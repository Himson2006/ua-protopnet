"""
Uncertainty scoring for Uncertainty-Aware ProtoPNet (UA-ProtoPNet).

These functions turn the prototype-layer *similarity* scores of a ProtoPNet
into interpretable uncertainty signals. The central idea: a ProtoPNet allocates
K prototypes per class, so the per-class evidence is the strongest prototype
activation within that class. When two *different* classes have nearly equal
top evidence, the model is genuinely torn between them — that competition is
both a scalar uncertainty signal *and* a pointer to the specific prototypes
that explain the uncertainty.

Input convention
----------------
``similarity_scores`` has shape ``[batch, num_prototypes]`` and is the prototype
*activation* (similarity, higher = more similar), i.e. what the original
ProtoPNet computes as ``model.distance_2_similarity(min_distances)``. Prototypes
are laid out class-contiguously::

    [c0_p0, c0_p1, ..., c0_p(K-1),  c1_p0, ..., c(C-1)_p(K-1)]

so a prototype's global index is ``class_id * K + local_idx`` and reshaping to
``[batch, C, K]`` recovers the per-class blocks. This matches ProtoPNet's
``prototype_class_identity[j, j // K] = 1`` allocation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Class evidence
# --------------------------------------------------------------------------- #

def compute_class_evidence(
    similarity_scores: torch.Tensor,
    num_classes: int,
    prototypes_per_class: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate per-prototype similarities into per-class evidence.

    For each class, the evidence is the **maximum** similarity among that
    class's K prototypes — i.e. how strongly *any* prototype of the class fires
    on the input.

    Parameters
    ----------
    similarity_scores : torch.Tensor
        Prototype activations, shape ``[batch, num_prototypes]`` (or
        ``[num_prototypes]`` for a single sample, which is promoted to a batch
        of 1).
    num_classes : int
        Number of classes ``C``.
    prototypes_per_class : int, optional
        ``K``. If omitted, inferred as ``num_prototypes // num_classes`` (which
        requires the prototypes to be evenly allocated, as in vanilla ProtoPNet).

    Returns
    -------
    torch.Tensor
        Class evidence, shape ``[batch, num_classes]`` (or ``[num_classes]`` if
        the input was 1-D).

    Notes
    -----
    Assumes the class-contiguous prototype layout described in the module
    docstring. The grouping is ``reshape([batch, C, K]).max(dim=2)``.
    """
    squeeze_back = False
    if similarity_scores.dim() == 1:
        similarity_scores = similarity_scores.unsqueeze(0)
        squeeze_back = True
    if similarity_scores.dim() != 2:
        raise ValueError(
            f"similarity_scores must be 1-D or 2-D, got shape "
            f"{tuple(similarity_scores.shape)}")

    batch, num_prototypes = similarity_scores.shape
    if prototypes_per_class is None:
        if num_prototypes % num_classes != 0:
            raise ValueError(
                f"num_prototypes ({num_prototypes}) is not divisible by "
                f"num_classes ({num_classes}); pass prototypes_per_class "
                f"explicitly for an uneven allocation.")
        prototypes_per_class = num_prototypes // num_classes

    expected = num_classes * prototypes_per_class
    if expected != num_prototypes:
        raise ValueError(
            f"num_classes*prototypes_per_class ({expected}) != num_prototypes "
            f"({num_prototypes}).")

    grouped = similarity_scores.view(batch, num_classes, prototypes_per_class)
    class_evidence = grouped.max(dim=2).values  # [batch, num_classes]
    return class_evidence.squeeze(0) if squeeze_back else class_evidence


# --------------------------------------------------------------------------- #
# Entropy-based uncertainty
# --------------------------------------------------------------------------- #

def compute_uncertainty_entropy(
    class_evidence: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Normalized Shannon entropy of the class-evidence distribution.

    Steps: softmax of ``class_evidence / T`` -> Shannon entropy -> divide by
    ``log(num_classes)`` so the result lies in ``[0, 1]``.

    Parameters
    ----------
    class_evidence : torch.Tensor
        Shape ``[batch, num_classes]`` (or ``[num_classes]`` for one sample).
    temperature : float
        Softmax temperature ``T``. ``T>1`` softens the distribution (more
        uncertain), ``T<1`` sharpens it. Calibrated on the val set elsewhere.

    Returns
    -------
    torch.Tensor
        Per-sample uncertainty in ``[0, 1]``: 0 = perfectly certain (one class
        dominates), 1 = maximally uncertain (uniform over classes). Shape
        ``[batch]`` (or scalar for a single sample).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    squeeze_back = False
    if class_evidence.dim() == 1:
        class_evidence = class_evidence.unsqueeze(0)
        squeeze_back = True

    num_classes = class_evidence.shape[1]
    log_probs = F.log_softmax(class_evidence / temperature, dim=1)
    probs = log_probs.exp()
    # Shannon entropy H = -sum p*log p ; p*log p -> 0 as p -> 0 (no NaN here
    # since log_softmax is finite).
    entropy = -(probs * log_probs).sum(dim=1)
    # Normalize by log(C); guard the degenerate single-class case.
    denom = torch.log(torch.tensor(float(num_classes),
                                   device=class_evidence.device))
    if num_classes <= 1:
        norm_entropy = torch.zeros_like(entropy)
    else:
        norm_entropy = (entropy / denom).clamp(0.0, 1.0)
    return norm_entropy.squeeze(0) if squeeze_back else norm_entropy


# --------------------------------------------------------------------------- #
# Competing prototypes (the interpretable core)
# --------------------------------------------------------------------------- #

def get_competing_prototypes(
    similarity_scores: torch.Tensor,
    num_classes: int,
    prototypes_per_class: int,
    class_names: Optional[Sequence[str]] = None,
    top_k_classes: int = 2,
) -> List[dict]:
    """Find the top competing prototypes from *different* classes (one sample).

    This is the interpretable heart of UA-ProtoPNet's uncertainty explanation.
    For a single sample, we compute each class's winning prototype (the most
    similar one), rank classes by that winning similarity, and return the top
    ``top_k_classes`` *distinct* classes. When the top-2 similarities are close,
    those two prototypes are "competing" to explain the image.

    Parameters
    ----------
    similarity_scores : torch.Tensor
        Prototype activations for ONE sample, shape ``[num_prototypes]`` (a
        ``[1, num_prototypes]`` tensor is also accepted).
    num_classes : int
        Number of classes ``C``.
    prototypes_per_class : int
        ``K`` prototypes per class.
    class_names : sequence of str, optional
        Human-readable class names; if given, each dict includes ``class_name``.
    top_k_classes : int
        How many distinct competing classes to return (default 2).

    Returns
    -------
    list of dict
        Ordered most- to least-similar, each with::

            {
              'rank':             int,    # 0 = winner
              'class_id':         int,
              'proto_local_idx':  int,    # index within the class block
              'proto_global_idx': int,    # class_id*K + local_idx
              'similarity_score': float,
              'class_name':       str,    # only if class_names provided
            }
    """
    if similarity_scores.dim() == 2 and similarity_scores.shape[0] == 1:
        similarity_scores = similarity_scores.squeeze(0)
    if similarity_scores.dim() != 1:
        raise ValueError(
            "get_competing_prototypes operates on a SINGLE sample; expected a "
            f"1-D tensor, got shape {tuple(similarity_scores.shape)}.")
    if top_k_classes > num_classes:
        top_k_classes = num_classes

    grouped = similarity_scores.view(num_classes, prototypes_per_class)
    # Winning prototype per class.
    per_class_max, per_class_argmax = grouped.max(dim=1)  # [C], [C]
    # Rank classes by their winning similarity, descending.
    order = torch.argsort(per_class_max, descending=True)

    results: List[dict] = []
    for rank in range(top_k_classes):
        class_id = int(order[rank].item())
        local_idx = int(per_class_argmax[class_id].item())
        entry = {
            "rank": rank,
            "class_id": class_id,
            "proto_local_idx": local_idx,
            "proto_global_idx": class_id * prototypes_per_class + local_idx,
            "similarity_score": float(per_class_max[class_id].item()),
        }
        if class_names is not None:
            entry["class_name"] = class_names[class_id]
        results.append(entry)
    return results


# --------------------------------------------------------------------------- #
# Competition score (alternative uncertainty measure)
# --------------------------------------------------------------------------- #

def compute_prototype_competition_score(
    class_evidence: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Top-1 vs top-2 class-evidence competition score.

    Defined per sample as::

        competition = 1 - (e_top1 - e_top2) / (e_top1 + eps)

    High competition (-> 1): the two strongest classes have nearly equal
    evidence => uncertain. Low competition (-> 0): one class dominates =>
    certain. This is an alternative to :func:`compute_uncertainty_entropy`; the
    paper reports and compares both.

    Parameters
    ----------
    class_evidence : torch.Tensor
        Shape ``[batch, num_classes]`` (or ``[num_classes]`` for one sample).
        Assumed non-negative (ProtoPNet 'log'/'linear' similarities are; if you
        feed signed evidence the score is still well-defined but less
        interpretable).
    eps : float
        Numerical floor on the denominator.

    Returns
    -------
    torch.Tensor
        Per-sample competition score, shape ``[batch]`` (scalar for one sample).
        Clamped to ``[0, 1]``.
    """
    squeeze_back = False
    if class_evidence.dim() == 1:
        class_evidence = class_evidence.unsqueeze(0)
        squeeze_back = True
    if class_evidence.shape[1] < 2:
        # Only one class: no competition possible.
        out = torch.zeros(class_evidence.shape[0], device=class_evidence.device)
        return out.squeeze(0) if squeeze_back else out

    top2 = torch.topk(class_evidence, k=2, dim=1).values  # [batch, 2]
    e_top1, e_top2 = top2[:, 0], top2[:, 1]
    competition = 1.0 - (e_top1 - e_top2) / (e_top1 + eps)
    competition = competition.clamp(0.0, 1.0)
    return competition.squeeze(0) if squeeze_back else competition


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    """Sanity-check shapes and the certain-vs-uncertain ordering on toy data."""
    torch.manual_seed(0)
    C, K, B = 4, 3, 5
    P = C * K

    # Construct a CERTAIN sample (class 0 dominant) and an UNCERTAIN sample
    # (classes 0 and 2 tie), plus random ones.
    sims = torch.rand(B, P) * 0.5
    sims[0, 0 * K + 1] = 5.0                       # certain -> class 0
    sims[1, 0 * K + 0] = 4.0; sims[1, 2 * K + 2] = 3.9  # uncertain: 0 vs 2

    evidence = compute_class_evidence(sims, num_classes=C)
    assert evidence.shape == (B, C), evidence.shape

    ent = compute_uncertainty_entropy(evidence, temperature=1.0)
    assert ent.shape == (B,) and (ent >= 0).all() and (ent <= 1).all()
    assert ent[1] > ent[0], (
        f"uncertain sample should have higher entropy: "
        f"certain={ent[0]:.3f} uncertain={ent[1]:.3f}")

    comp = compute_prototype_competition_score(evidence)
    assert comp.shape == (B,) and (comp >= 0).all() and (comp <= 1).all()
    assert comp[1] > comp[0], "uncertain sample should have higher competition"

    names = [f"class_{i}" for i in range(C)]
    competitors = get_competing_prototypes(
        sims[1], num_classes=C, prototypes_per_class=K, class_names=names)
    assert len(competitors) == 2
    assert competitors[0]["class_id"] != competitors[1]["class_id"]
    assert {competitors[0]["class_id"], competitors[1]["class_id"]} == {0, 2}, \
        competitors

    # Single-sample (1-D) path for class evidence + entropy.
    ev1 = compute_class_evidence(sims[1], num_classes=C)
    assert ev1.shape == (C,)
    e1 = compute_uncertainty_entropy(ev1)
    assert e1.dim() == 0

    print("certain   sample: entropy=%.3f competition=%.3f" % (ent[0], comp[0]))
    print("uncertain sample: entropy=%.3f competition=%.3f" % (ent[1], comp[1]))
    print("competing prototypes (uncertain sample):")
    for c in competitors:
        print("  rank %d: %s  global_idx=%d  sim=%.3f"
              % (c["rank"], c["class_name"], c["proto_global_idx"],
                 c["similarity_score"]))
    print("[uncertainty_scores] self-test OK")


if __name__ == "__main__":
    _self_test()
