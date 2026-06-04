"""
Evaluation metrics for UA-ProtoPNet.

Implements calibration (ECE), uncertainty quality (AUROC of uncertainty vs.
error), the ambiguous-vs-clear uncertainty ratio, and — the headline LIDC-IDRI
metric — the correlation between model uncertainty and inter-radiologist
disagreement. :func:`run_full_evaluation` runs one pass over a dataloader and
returns every metric in a single dict, so the same routine works for the main
model and the baselines (Section 7).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .pp_compat import forward_with_similarity, get_prototypes_per_class, unwrap
from .uncertainty_scores import (
    compute_class_evidence,
    compute_prototype_competition_score,
    compute_uncertainty_entropy,
)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (equal-width binning).

    Partitions ``[0, 1]`` into ``n_bins`` confidence bins; for each bin computes
    ``|mean(confidence) - mean(accuracy)|`` weighted by the fraction of samples
    in the bin, and sums.

    Parameters
    ----------
    confidences : array-like, shape [N]
        Predicted confidence (e.g. max softmax probability) per sample, in [0,1].
    accuracies : array-like, shape [N]
        1.0 if that prediction was correct else 0.0.
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    float
        ECE in ``[0, 1]`` (lower is better).
    """
    confidences = np.asarray(confidences, dtype=np.float64).ravel()
    accuracies = np.asarray(accuracies, dtype=np.float64).ravel()
    if confidences.size == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.size
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Last bin is closed on the right so confidence == 1.0 is included.
        if hi >= 1.0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = accuracies[in_bin].mean()
        ece += (count / n) * abs(avg_conf - avg_acc)
    return float(ece)


# --------------------------------------------------------------------------- #
# Uncertainty quality
# --------------------------------------------------------------------------- #

def uncertainty_auroc(
    uncertainty_scores: np.ndarray,
    is_incorrect: np.ndarray,
) -> float:
    """AUROC for using uncertainty to predict misclassification.

    Treats ``is_incorrect`` (1 = wrong prediction) as the positive class and the
    uncertainty score as the ranking score. A good uncertainty estimator scores
    > 0.70: errors tend to carry higher uncertainty than correct predictions.

    Implemented from scratch (rank-based / Mann-Whitney U) so it has no hard
    sklearn dependency and degrades gracefully when only one class is present.

    Returns
    -------
    float
        AUROC in ``[0, 1]``; ``nan`` if all predictions are correct or all wrong
        (AUROC undefined).
    """
    u = np.asarray(uncertainty_scores, dtype=np.float64).ravel()
    y = np.asarray(is_incorrect).astype(bool).ravel()
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank-based AUROC = (sum of ranks of positives - n_pos*(n_pos+1)/2) /
    # (n_pos * n_neg), with average ranks for ties.
    order = np.argsort(u, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(u) + 1)
    # Resolve ties to average rank.
    _, inv, counts = np.unique(u, return_inverse=True, return_counts=True)
    # Sum of ranks within each unique value, redistributed as the average rank.
    rank_sum = np.zeros_like(counts, dtype=np.float64)
    np.add.at(rank_sum, inv, ranks)
    avg_rank = rank_sum / counts
    ranks = avg_rank[inv]

    sum_ranks_pos = ranks[y].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def uncertainty_on_ambiguous_vs_clear(
    uncertainty_scores: np.ndarray,
    labels: np.ndarray,
    ambiguous_label,
) -> Dict[str, float]:
    """Compare mean uncertainty on ambiguous vs. clear samples.

    Parameters
    ----------
    uncertainty_scores : array-like, shape [N]
        Per-sample uncertainty.
    labels : array-like, shape [N]
        Integer labels.
    ambiguous_label : int or sequence of int or bool-array
        Which samples are "ambiguous". May be a single class index, a collection
        of class indices, or a precomputed boolean mask of shape [N].

    Returns
    -------
    dict
        ``{'u_ambiguous', 'u_clear', 'ratio', 'n_ambiguous', 'n_clear'}``.
        ``ratio = u_ambiguous / u_clear`` (>1.5 good, >2.0 strong).
    """
    u = np.asarray(uncertainty_scores, dtype=np.float64).ravel()
    labels = np.asarray(labels)

    if np.asarray(ambiguous_label).dtype == bool and \
            np.asarray(ambiguous_label).shape == u.shape:
        amb_mask = np.asarray(ambiguous_label, dtype=bool)
    else:
        amb_ids = np.atleast_1d(np.asarray(ambiguous_label))
        amb_mask = np.isin(labels, amb_ids)
    clear_mask = ~amb_mask

    u_amb = float(u[amb_mask].mean()) if amb_mask.any() else float("nan")
    u_clear = float(u[clear_mask].mean()) if clear_mask.any() else float("nan")
    ratio = (u_amb / u_clear) if (u_clear and np.isfinite(u_clear) and
                                  u_clear != 0) else float("nan")
    return {
        "u_ambiguous": u_amb,
        "u_clear": u_clear,
        "ratio": ratio,
        "n_ambiguous": int(amb_mask.sum()),
        "n_clear": int(clear_mask.sum()),
    }


# --------------------------------------------------------------------------- #
# LIDC-IDRI headline metric
# --------------------------------------------------------------------------- #

def correlation_with_radiologist_std(
    uncertainty_scores: np.ndarray,
    radiologist_std_scores: np.ndarray,
) -> Dict[str, float]:
    """Pearson correlation between model uncertainty and inter-radiologist std.

    LIDC-IDRI only. The std of the four radiologists' malignancy ratings is a
    ground-truth aleatoric-uncertainty signal; a strong positive correlation
    (r >= 0.4, even 0.3 is publishable) is the key novelty result.

    Returns
    -------
    dict
        ``{'pearson_r', 'p_value', 'n_samples'}``. Uses ``scipy.stats.pearsonr``
        when available; otherwise falls back to a numpy correlation with a
        normal-approximation p-value.
    """
    u = np.asarray(uncertainty_scores, dtype=np.float64).ravel()
    s = np.asarray(radiologist_std_scores, dtype=np.float64).ravel()
    mask = np.isfinite(u) & np.isfinite(s)
    u, s = u[mask], s[mask]
    n = u.size
    if n < 3 or u.std() == 0 or s.std() == 0:
        return {"pearson_r": float("nan"), "p_value": float("nan"),
                "n_samples": int(n)}
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(u, s)
    except Exception:
        r = float(np.corrcoef(u, s)[0, 1])
        # t-approximation for the two-sided p-value.
        import math
        t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
        # Survival function of |t| via error function approximation.
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"pearson_r": float(r), "p_value": float(p), "n_samples": int(n)}


# --------------------------------------------------------------------------- #
# Full evaluation pass
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_predictions(
    model,
    dataloader,
    num_classes: int,
    K: int,
    device,
    temperature: float = 1.0,
    forward_fn: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """Run the model over a dataloader and collect per-sample arrays.

    Works with dataloaders that yield ``(image, label)`` (HAM10000) or
    ``(image, hard_label, soft_label, std)`` (LIDC-IDRI); extra elements are
    captured when present.

    ``forward_fn(model, images)`` may be supplied to support baselines that have
    a non-standard forward (e.g. MC-Dropout averaging). It must return
    ``(logits, class_evidence, uncertainty)`` where any of the latter two may be
    ``None`` to fall back to the standard prototype-similarity computation.

    Returns a dict of numpy arrays: ``logits, probs, preds, labels, confidences,
    accuracies, is_incorrect, entropy, competition, class_evidence`` and, if the
    loader provides it, ``radiologist_std``.
    """
    model.eval()
    out: Dict[str, List] = {k: [] for k in (
        "preds", "labels", "confidences", "accuracies", "is_incorrect",
        "entropy", "competition", "radiologist_std", "class_evidence")}

    for batch in dataloader:
        images = batch[0].to(device)
        labels = batch[1]
        rad_std = batch[3] if len(batch) >= 4 else None

        if forward_fn is not None:
            logits, class_evidence, uncertainty = forward_fn(model, images)
        else:
            logits, _min_d, sims = forward_with_similarity(model, images)
            class_evidence = compute_class_evidence(sims, num_classes, K)
            uncertainty = None

        if class_evidence is None:
            _logits2, _m, sims = forward_with_similarity(model, images)
            class_evidence = compute_class_evidence(sims, num_classes, K)

        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        preds = preds.cpu()
        labels_cpu = labels.cpu()
        correct = (preds == labels_cpu)

        entropy = compute_uncertainty_entropy(class_evidence, temperature)
        competition = compute_prototype_competition_score(class_evidence)
        if uncertainty is None:
            uncertainty = entropy  # default scalar uncertainty = entropy

        out["preds"].append(preds.numpy())
        out["labels"].append(labels_cpu.numpy())
        out["confidences"].append(conf.cpu().numpy())
        out["accuracies"].append(correct.numpy().astype(np.float64))
        out["is_incorrect"].append((~correct).numpy().astype(np.float64))
        out["entropy"].append(np.asarray(uncertainty.cpu()).ravel())
        out["competition"].append(competition.cpu().numpy().ravel())
        out["class_evidence"].append(class_evidence.cpu().numpy())
        if rad_std is not None:
            out["radiologist_std"].append(np.asarray(rad_std).ravel())

    if not out["labels"]:
        raise ValueError(
            "collect_predictions received an empty dataloader (no batches). "
            "Check the split sizes / max_samples.")

    result: Dict[str, np.ndarray] = {}
    for k, v in out.items():
        if len(v) == 0:
            continue
        result[k] = (np.concatenate(v) if v[0].ndim >= 1 and k != "class_evidence"
                     else np.concatenate(v, axis=0))
    return result


def run_full_evaluation(
    model,
    dataloader,
    num_classes: int,
    K: int,
    device,
    radiologist_std: Optional[np.ndarray] = None,
    ambiguous_label=None,
    temperature: float = 1.0,
    forward_fn: Optional[Callable] = None,
    n_ece_bins: int = 15,
) -> Dict[str, object]:
    """Full evaluation pass returning a comprehensive metrics dict.

    Parameters
    ----------
    model, dataloader, num_classes, K, device
        Standard arguments. ``K`` = prototypes per class.
    radiologist_std : array-like, optional
        Per-sample inter-radiologist std (LIDC). If None, taken from the loader
        when it yields a 4-tuple.
    ambiguous_label : int / sequence / bool-mask, optional
        Defines the ambiguous group for the ambiguous-vs-clear metric. For LIDC
        this is class 1; for HAM10000 pass the ambiguous class indices.
    temperature : float
        Calibrated softmax temperature applied to class evidence.
    forward_fn : callable, optional
        Custom forward for baselines (see :func:`collect_predictions`).

    Returns
    -------
    dict
        ``accuracy, ece, uncertainty_auroc, ambiguous_vs_clear (dict),
        mean_uncertainty, temperature`` and, when radiologist std is available,
        ``radiologist_correlation`` (dict). Also echoes ``n_samples``.
    """
    pred = collect_predictions(model, dataloader, num_classes, K, device,
                               temperature=temperature, forward_fn=forward_fn)

    accuracy = float(pred["accuracies"].mean())
    ece = compute_ece(pred["confidences"], pred["accuracies"], n_bins=n_ece_bins)
    auroc = uncertainty_auroc(pred["entropy"], pred["is_incorrect"])

    metrics: Dict[str, object] = {
        "n_samples": int(pred["labels"].size),
        "accuracy": accuracy,
        "ece": ece,
        "uncertainty_auroc": auroc,
        "mean_uncertainty": float(pred["entropy"].mean()),
        "mean_competition": float(pred["competition"].mean()),
        "temperature": float(temperature),
    }

    if ambiguous_label is not None:
        metrics["ambiguous_vs_clear"] = uncertainty_on_ambiguous_vs_clear(
            pred["entropy"], pred["labels"], ambiguous_label)

    std = radiologist_std
    if std is None and "radiologist_std" in pred:
        std = pred["radiologist_std"]
    if std is not None:
        metrics["radiologist_correlation"] = correlation_with_radiologist_std(
            pred["entropy"], std)

    return metrics


# --------------------------------------------------------------------------- #
# Temperature calibration helper (Section 10)
# --------------------------------------------------------------------------- #

def calibrate_temperature(
    model,
    val_loader,
    num_classes: int,
    K: int,
    device,
    candidates: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    n_ece_bins: int = 15,
) -> float:
    """Select the temperature in ``candidates`` minimizing validation ECE.

    Confidence for ECE is taken as the max softmax probability of the
    *temperature-scaled class evidence* (so the swept T actually affects the
    calibration objective). Returns the best T.
    """
    pred = collect_predictions(model, val_loader, num_classes, K, device,
                               temperature=1.0)
    evidence = torch.tensor(pred["class_evidence"], dtype=torch.float32)
    labels = torch.tensor(pred["labels"])
    best_T, best_ece = 1.0, float("inf")
    for T in candidates:
        probs = F.softmax(evidence / T, dim=1)
        conf, preds = probs.max(dim=1)
        acc = (preds == labels).numpy().astype(np.float64)
        ece = compute_ece(conf.numpy(), acc, n_bins=n_ece_bins)
        if ece < best_ece:
            best_ece, best_T = ece, float(T)
    return best_T


# --------------------------------------------------------------------------- #
# Self-test (synthetic, no model needed)
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    rng = np.random.RandomState(0)
    n = 500
    # Well-calibrated-ish: confidence correlates with correctness.
    conf = rng.uniform(0.5, 1.0, n)
    acc = (rng.uniform(size=n) < conf).astype(float)
    ece = compute_ece(conf, acc)
    assert 0.0 <= ece <= 1.0

    # Uncertainty higher on errors -> AUROC > 0.5.
    is_wrong = (1 - acc).astype(bool)
    unc = np.where(is_wrong, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    auroc = uncertainty_auroc(unc, is_wrong)
    assert auroc > 0.8, auroc

    labels = rng.randint(0, 3, n)
    unc2 = np.where(labels == 1, rng.uniform(0.6, 1.0, n),
                    rng.uniform(0.0, 0.4, n))
    avc = uncertainty_on_ambiguous_vs_clear(unc2, labels, ambiguous_label=1)
    assert avc["ratio"] > 1.5, avc

    std = rng.uniform(0, 1.2, n)
    corr = correlation_with_radiologist_std(0.7 * std + 0.1 * rng.randn(n), std)
    assert corr["pearson_r"] > 0.5, corr

    print("ECE=%.4f  AUROC=%.3f  amb/clear ratio=%.2f  pearson_r=%.3f" %
          (ece, auroc, avc["ratio"], corr["pearson_r"]))
    print("[evaluate] self-test OK")


if __name__ == "__main__":
    _self_test()
