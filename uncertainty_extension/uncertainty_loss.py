"""
Uncertainty-aware loss terms for UA-ProtoPNet.

These augment the standard ProtoPNet objective (cross-entropy + cluster +
separation + L1) with terms that (a) *calibrate* uncertainty — forcing high
entropy on ambiguous inputs and low entropy on clear ones, (b) support
soft-label supervision for LIDC-IDRI radiologist vote distributions, and
(c) keep prototypes diverse so that "competing prototype" explanations remain
meaningful.

All functions return scalar tensors suitable for summation into a total loss.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .uncertainty_scores import compute_uncertainty_entropy


# --------------------------------------------------------------------------- #
# Uncertainty calibration
# --------------------------------------------------------------------------- #

def uncertainty_calibration_loss(
    class_evidence: torch.Tensor,
    labels: torch.Tensor,
    ambiguous_mask: torch.Tensor,
    temperature: float = 1.0,
    enforce_certainty_on_clear: bool = True,
    clear_weight: float = 1.0,
) -> torch.Tensor:
    """Push the model to be uncertain on ambiguous inputs and certain on clear.

    Uses the normalized class-evidence entropy in ``[0, 1]`` (see
    :func:`uncertainty_scores.compute_uncertainty_entropy`).

    * **Ambiguous samples** (``ambiguous_mask == True``): we want *high*
      entropy, so we add ``-mean(entropy[ambiguous])``. Minimizing the total
      loss therefore maximizes their entropy.
    * **Clear samples** (``ambiguous_mask == False``), only if
      ``enforce_certainty_on_clear``: we want *low* entropy, so we add
      ``+clear_weight * mean(entropy[clear])``.

    Empty groups contribute 0 (a batch with no ambiguous samples is fine).

    Parameters
    ----------
    class_evidence : torch.Tensor
        ``[batch, num_classes]`` per-class evidence.
    labels : torch.Tensor
        ``[batch]`` hard labels. Currently unused by the loss itself but kept in
        the signature (per spec) for symmetry / future label-conditioned terms.
    ambiguous_mask : torch.Tensor
        ``[batch]`` boolean; ``True`` marks ambiguous-class samples.
    temperature : float
        Softmax temperature used inside the entropy computation.
    enforce_certainty_on_clear : bool
        Whether to additionally penalize high entropy on clear samples.
    clear_weight : float
        Relative weight of the clear-sample certainty term.

    Returns
    -------
    torch.Tensor
        Scalar combined calibration loss (can be negative — that is expected,
        since the ambiguous term is ``-entropy``).
    """
    del labels  # reserved for future use; mask already encodes the grouping
    ambiguous_mask = ambiguous_mask.to(torch.bool)
    entropy = compute_uncertainty_entropy(class_evidence, temperature=temperature)

    zero = class_evidence.new_zeros(())
    # Ambiguous: maximize entropy  ->  minimize (-entropy).
    if ambiguous_mask.any():
        loss_ambiguous = -entropy[ambiguous_mask].mean()
    else:
        loss_ambiguous = zero

    # Clear: minimize entropy.
    loss_clear = zero
    if enforce_certainty_on_clear:
        clear_mask = ~ambiguous_mask
        if clear_mask.any():
            loss_clear = clear_weight * entropy[clear_mask].mean()

    return loss_ambiguous + loss_clear


# --------------------------------------------------------------------------- #
# Soft-label cross-entropy (LIDC-IDRI)
# --------------------------------------------------------------------------- #

def soft_label_cross_entropy(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy against soft (probability-vector) targets.

    Used for LIDC-IDRI, where each nodule's target is the radiologist vote
    distribution ``[frac_benign, frac_uncertain, frac_malignant]`` rather than a
    single hard label. Reduces to standard CE when ``soft_labels`` is one-hot.

    Formula: ``-mean( sum_c soft_labels[:, c] * log_softmax(logits)[:, c] )``.

    Parameters
    ----------
    logits : torch.Tensor
        ``[batch, num_classes]`` raw class logits (here, class evidence/logits).
    soft_labels : torch.Tensor
        ``[batch, num_classes]`` non-negative rows that (ideally) sum to 1.

    Returns
    -------
    torch.Tensor
        Scalar mean soft-label cross-entropy.
    """
    if logits.shape != soft_labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and soft_labels "
            f"{tuple(soft_labels.shape)} must have the same shape.")
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# Prototype diversity
# --------------------------------------------------------------------------- #

def prototype_diversity_loss(
    prototype_vectors: torch.Tensor,
    prototype_class_identity: torch.Tensor,
    same_class_margin: float = 0.5,
    cross_class_margin: float = 0.0,
    same_class_weight: float = 1.0,
    cross_class_weight: float = 1.0,
    separate_cross_class: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Keep prototypes diverse: cross-class apart, same-class non-collapsed.

    Works on cosine similarity between flattened prototype vectors.

    * **Same-class collapse** (always on): penalize pairs of same-class
      prototypes whose cosine similarity exceeds ``same_class_margin``
      (``relu(cos - margin)``). This prevents a class's K prototypes from
      degenerating into K copies of one pattern.
    * **Cross-class separation** (on iff ``separate_cross_class``): penalize
      pairs of different-class prototypes whose cosine similarity exceeds
      ``cross_class_margin``, pushing different classes' prototypes apart so the
      "competing prototype" explanation is grounded in genuinely distinct
      patterns.

    Parameters
    ----------
    prototype_vectors : torch.Tensor
        ``[P, D, 1, 1]`` (ProtoPNet layout) or ``[P, D]``.
    prototype_class_identity : torch.Tensor
        ``[P, num_classes]`` one-hot (ProtoPNet's ``prototype_class_identity``).
    same_class_margin, cross_class_margin : float
        Cosine-similarity margins above which pairs are penalized.
    same_class_weight, cross_class_weight : float
        Relative weights of the two terms.
    separate_cross_class : bool
        If False, only the same-class collapse term is used.
    eps : float
        Numerical floor for normalization.

    Returns
    -------
    torch.Tensor
        Scalar diversity loss (>= 0).
    """
    P = prototype_vectors.shape[0]
    v = prototype_vectors.reshape(P, -1)
    v = F.normalize(v, dim=1, eps=eps)
    cos = v @ v.t()  # [P, P] in [-1, 1]

    class_id = prototype_class_identity.argmax(dim=1).to(cos.device)  # [P]
    same = class_id.unsqueeze(0) == class_id.unsqueeze(1)             # [P, P]
    eye = torch.eye(P, dtype=torch.bool, device=cos.device)
    same_off = same & ~eye   # same class, distinct prototypes
    cross = ~same            # different classes

    loss = cos.new_zeros(())

    if same_off.any():
        loss = loss + same_class_weight * \
            F.relu(cos[same_off] - same_class_margin).mean()

    if separate_cross_class and cross.any():
        loss = loss + cross_class_weight * \
            F.relu(cos[cross] - cross_class_margin).mean()

    return loss


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    torch.manual_seed(0)
    C, K, B = 3, 4, 6
    P = C * K

    # --- calibration loss ---------------------------------------------------
    evidence = torch.rand(B, C)
    # Make first 3 ambiguous (flat evidence), last 3 clear (peaked).
    evidence[:3] = 0.5
    evidence[3:] = 0.1
    evidence[3:, 0] = 5.0
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    mask = torch.tensor([True, True, True, False, False, False])
    cal = uncertainty_calibration_loss(evidence, labels, mask,
                                       enforce_certainty_on_clear=True)
    assert cal.dim() == 0
    # With no ambiguous samples the term must vanish gracefully.
    cal_none = uncertainty_calibration_loss(
        evidence, labels, torch.zeros(B, dtype=torch.bool))
    assert torch.isfinite(cal_none)

    # --- soft-label CE ------------------------------------------------------
    logits = torch.randn(B, C)
    soft = torch.softmax(torch.randn(B, C), dim=1)
    sce = soft_label_cross_entropy(logits, soft)
    # One-hot soft labels should match standard CE.
    hard = torch.tensor([0, 1, 2, 0, 1, 2])
    onehot = F.one_hot(hard, C).float()
    sce_oh = soft_label_cross_entropy(logits, onehot)
    ce_ref = F.cross_entropy(logits, hard)
    assert torch.allclose(sce_oh, ce_ref, atol=1e-5), (sce_oh, ce_ref)

    # --- prototype diversity ------------------------------------------------
    pci = torch.zeros(P, C)
    for j in range(P):
        pci[j, j // K] = 1.0
    # Collapsed prototypes (all identical) -> high loss.
    collapsed = torch.ones(P, 8, 1, 1)
    loss_collapsed = prototype_diversity_loss(collapsed, pci)
    # Diverse (orthogonal-ish) prototypes -> lower loss.
    diverse = torch.eye(P)[:, :8].reshape(P, 8, 1, 1) + 0.01 * torch.randn(P, 8, 1, 1)
    loss_diverse = prototype_diversity_loss(diverse, pci)
    assert loss_collapsed > loss_diverse, (loss_collapsed, loss_diverse)
    assert loss_diverse >= 0

    print("calibration loss (amb+clear):      %.4f" % cal.item())
    print("soft-CE one-hot matches CE:        %.6f vs %.6f" % (sce_oh, ce_ref))
    print("diversity collapsed vs diverse:    %.4f > %.4f" %
          (loss_collapsed.item(), loss_diverse.item()))
    print("[uncertainty_loss] self-test OK")


if __name__ == "__main__":
    _self_test()
