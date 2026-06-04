"""
Visualization for UA-ProtoPNet (Section 6).

Three figures:

* :func:`visualize_uncertainty_explanation` — the headline 4-panel figure for a
  single uncertain sample: input + competing-prototype heatmaps, the two winning
  prototype patches (loaded from the push directory), and a class-evidence bar
  chart annotated with the uncertainty score.
* :func:`plot_uncertainty_distribution` — per-class uncertainty distributions
  (violin), with the clear-class median drawn as a reference line.
* :func:`plot_correlation_scatter` — (LIDC only) model uncertainty vs.
  inter-radiologist std, colored by hard label, with a regression line and the
  Pearson r / p-value in the title.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe (servers / CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .pp_compat import forward_with_similarity, get_prototypes_per_class, unwrap  # noqa: E402
from .uncertainty_scores import (  # noqa: E402
    compute_class_evidence,
    compute_uncertainty_entropy,
    get_competing_prototypes,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization -> HxWx3 float image in [0,1] for display."""
    img = image_tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


@torch.no_grad()
def _prototype_activation_map(model, image_tensor, proto_global_idx, out_size):
    """Upsampled spatial similarity map for one prototype over the input."""
    base = unwrap(model)
    distances = base.prototype_distances(image_tensor)        # [1, P, h, w]
    sim = base.distance_2_similarity(distances)[0, proto_global_idx]  # [h, w]
    sim = sim.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(sim, size=(out_size, out_size), mode="bicubic",
                       align_corners=False)[0, 0].cpu().numpy()
    # Normalize to [0,1] for an alpha overlay.
    up = up - up.min()
    if up.max() > 0:
        up = up / up.max()
    return up


def _load_prototype_patch(prototype_image_dir, proto_global_idx,
                          prefix="prototype-img"):
    """Load a saved prototype patch png if present, else return None."""
    if prototype_image_dir is None:
        return None
    # The push may save under epoch-<N>/; search the dir tree for the patch.
    fname = f"{prefix}{proto_global_idx}.png"
    candidates = [os.path.join(prototype_image_dir, fname)]
    for root, _d, files in os.walk(prototype_image_dir):
        if fname in files:
            candidates.append(os.path.join(root, fname))
    for path in candidates:
        if os.path.isfile(path):
            try:
                return plt.imread(path)
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# The headline figure
# --------------------------------------------------------------------------- #

@torch.no_grad()
def visualize_uncertainty_explanation(
    model,
    image_tensor: torch.Tensor,
    true_label,
    class_names: Sequence[str],
    prototype_image_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    temperature: float = 1.0,
    device=None,
):
    """4-panel interpretable-uncertainty figure for one sample.

    Parameters
    ----------
    model : nn.Module
        Trained PPNet (post-push, so prototypes correspond to saved patches).
    image_tensor : torch.Tensor
        Single normalized image ``[3,224,224]`` or ``[1,3,224,224]``.
    true_label : int
        Ground-truth class index (for the title; -1 if unknown).
    class_names : sequence of str
    prototype_image_dir : str, optional
        The push save directory containing ``prototype-img<j>.png``.
    save_path : str, optional
        If given, the figure is written there (and closed).

    Returns
    -------
    matplotlib.figure.Figure
    """
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if device is not None:
        image_tensor = image_tensor.to(device)

    base = unwrap(model)
    num_classes = base.num_classes
    K = get_prototypes_per_class(model)

    logits, _min_d, sims = forward_with_similarity(model, image_tensor)
    class_evidence = compute_class_evidence(sims, num_classes, K)  # [1, C]
    uncertainty = float(compute_uncertainty_entropy(class_evidence, temperature)[0])
    pred = int(logits.argmax(dim=1)[0].item())

    competitors = get_competing_prototypes(
        sims[0], num_classes, K, class_names=list(class_names))
    top, second = competitors[0], competitors[1]

    disp_img = _denormalize(image_tensor)
    img_size = disp_img.shape[0]
    act_top = _prototype_activation_map(model, image_tensor,
                                        top["proto_global_idx"], img_size)
    act_2nd = _prototype_activation_map(model, image_tensor,
                                        second["proto_global_idx"], img_size)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    # Panel 1: input + both competing heatmaps.
    axes[0].imshow(disp_img)
    axes[0].imshow(np.dstack([np.ones_like(act_top), np.zeros_like(act_top),
                              np.zeros_like(act_top), 0.45 * act_top]))
    axes[0].imshow(np.dstack([np.zeros_like(act_2nd), np.zeros_like(act_2nd),
                              np.ones_like(act_2nd), 0.45 * act_2nd]))
    axes[0].set_title("Input + competing regions\n(red=%s, blue=%s)"
                      % (top["class_name"], second["class_name"]), fontsize=10)
    axes[0].axis("off")

    # Panels 2 & 3: each competitor's prototype patch + activation overlay.
    for ax, comp, cmap_color, overlay in (
        (axes[1], top, "Reds", act_top),
        (axes[2], second, "Blues", act_2nd),
    ):
        patch = _load_prototype_patch(prototype_image_dir, comp["proto_global_idx"])
        if patch is not None:
            ax.imshow(patch)
            ax.set_title("Prototype P%d [%s]\nsimilarity=%.2f"
                         % (comp["proto_global_idx"], comp["class_name"],
                            comp["similarity_score"]), fontsize=10)
        else:
            # No saved patch (e.g. pre-push / smoke test): show activation region.
            ax.imshow(disp_img)
            ax.imshow(overlay, cmap=cmap_color, alpha=0.5)
            ax.set_title("P%d [%s] sim=%.2f\n(patch png not found)"
                         % (comp["proto_global_idx"], comp["class_name"],
                            comp["similarity_score"]), fontsize=10)
        ax.axis("off")

    # Panel 4: class-evidence bar chart.
    evidence = class_evidence[0].cpu().numpy()
    colors = ["tab:gray"] * num_classes
    colors[top["class_id"]] = "tab:red"
    colors[second["class_id"]] = "tab:blue"
    axes[3].barh(range(num_classes), evidence, color=colors)
    axes[3].set_yticks(range(num_classes))
    axes[3].set_yticklabels(list(class_names), fontsize=8)
    axes[3].invert_yaxis()
    axes[3].set_xlabel("Class evidence (max prototype similarity)")
    axes[3].set_title("Class evidence", fontsize=10)
    axes[3].text(0.97, 0.04, f"U = {uncertainty:.3f}",
                 transform=axes[3].transAxes, ha="right", va="bottom",
                 fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))

    true_name = (class_names[true_label] if isinstance(true_label, int)
                 and 0 <= true_label < num_classes else str(true_label))
    fig.suptitle(
        f"Uncertain Prediction — True: {true_name} | Model: {class_names[pred]} "
        f"| U = {uncertainty:.3f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Distribution plot
# --------------------------------------------------------------------------- #

def plot_uncertainty_distribution(
    uncertainty_scores_by_class: Dict[str, np.ndarray],
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    clear_class_names: Optional[Sequence[str]] = None,
):
    """Violin plot of uncertainty distributions per class.

    Parameters
    ----------
    uncertainty_scores_by_class : dict
        ``{class_name: array_of_uncertainty_scores}``.
    class_names : sequence of str
        Ordering of classes along the x-axis.
    clear_class_names : sequence of str, optional
        Which classes count as "clear"; their pooled median is drawn as a
        dashed reference line. Defaults to all classes.
    """
    names = [c for c in class_names if c in uncertainty_scores_by_class]
    data = [np.asarray(uncertainty_scores_by_class[c], dtype=float) for c in names]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Uncertainty score U")
    ax.set_title("Uncertainty distribution by class")

    clear = clear_class_names if clear_class_names is not None else names
    pooled = np.concatenate(
        [np.asarray(uncertainty_scores_by_class[c], dtype=float)
         for c in clear if c in uncertainty_scores_by_class]) \
        if any(c in uncertainty_scores_by_class for c in clear) else np.array([])
    if pooled.size:
        med = float(np.median(pooled))
        ax.axhline(med, ls="--", color="gray",
                   label=f"clear-class median = {med:.3f}")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Correlation scatter (LIDC)
# --------------------------------------------------------------------------- #

def plot_correlation_scatter(
    model_uncertainty: np.ndarray,
    radiologist_std: np.ndarray,
    save_path: Optional[str] = None,
    hard_labels: Optional[np.ndarray] = None,
    label_names: Sequence[str] = ("benign", "uncertain", "malignant"),
):
    """Scatter of model uncertainty vs. inter-radiologist std (LIDC).

    Colors points by hard label when provided, adds a least-squares trend line,
    and reports Pearson r / p-value in the title.
    """
    u = np.asarray(model_uncertainty, dtype=float).ravel()
    s = np.asarray(radiologist_std, dtype=float).ravel()
    mask = np.isfinite(u) & np.isfinite(s)
    u, s = u[mask], s[mask]

    from .evaluate import correlation_with_radiologist_std
    corr = correlation_with_radiologist_std(u, s)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    if hard_labels is not None:
        hl = np.asarray(hard_labels).ravel()[mask]
        for cls in np.unique(hl):
            m = hl == cls
            name = label_names[int(cls)] if int(cls) < len(label_names) else str(cls)
            ax.scatter(s[m], u[m], alpha=0.6, s=18, label=name)
        ax.legend(title="hard label", fontsize=8)
    else:
        ax.scatter(s, u, alpha=0.6, s=18)

    # Least-squares trend line.
    if u.size >= 2 and s.std() > 0:
        b, a = np.polyfit(s, u, 1)
        xs = np.linspace(s.min(), s.max(), 100)
        ax.plot(xs, b * xs + a, color="black", lw=1.5, ls="--")

    ax.set_xlabel("Inter-radiologist std (ground-truth aleatoric uncertainty)")
    ax.set_ylabel("Model uncertainty U")
    ax.set_title("Model uncertainty vs. radiologist disagreement\n"
                 f"Pearson r = {corr['pearson_r']:.3f}, "
                 f"p = {corr['p_value']:.2e}  (n={corr['n_samples']})")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
