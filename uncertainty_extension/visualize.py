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
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .pp_compat import forward_with_similarity, get_prototypes_per_class, unwrap  # noqa: E402
from .uncertainty_scores import compute_uncertainty_entropy  # noqa: E402

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


def _activation_bbox(act, thresh=0.6):
    """Bounding box ``(x0, y0, x1, y1)`` of the high-activation region.

    ``act`` is a [0,1]-normalized upsampled similarity map; we threshold at
    ``thresh`` of its max and return the tight box around the resulting mask.
    """
    peak = float(act.max())
    if peak <= 0:
        return None
    ys, xs = np.where(act >= thresh * peak)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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


def _competing_prototypes_by_logits(model, logits_row, sims_row, num_classes, K,
                                    class_names):
    """Top-2 competing classes/prototypes grounded in the model's decision.

    The two competing classes are the top-2 by **logit** (what the model is
    actually deciding between). For each, the representative prototype is the one
    of that class contributing most to its logit, i.e. ``argmax_p activation_p *
    last_layer.weight[c, p]`` over the class's prototypes. This avoids the
    class-evidence failure where near-uniform max-similarities surface arbitrary,
    clinically irrelevant 'magnet' prototypes.
    """
    base = unwrap(model)
    W = base.last_layer.weight.detach().to(sims_row.device)   # [C, P]
    top2 = torch.topk(logits_row, k=min(2, num_classes)).indices.tolist()
    out = []
    for rank, c in enumerate(top2):
        start, end = c * K, (c + 1) * K
        contrib = sims_row[start:end] * W[c, start:end]       # per-prototype contribution
        local = int(torch.argmax(contrib).item())
        gidx = start + local
        out.append({
            "rank": rank, "class_id": int(c), "proto_local_idx": local,
            "proto_global_idx": int(gidx),
            "similarity_score": float(sims_row[gidx]),
            "class_name": class_names[c],
        })
    return out


def _load_proto_metadata(prototype_metadata):
    """Accept a path or a list; return {proto_global_idx: entry}."""
    if prototype_metadata is None:
        return {}
    meta = prototype_metadata
    if isinstance(meta, str) and os.path.isfile(meta):
        try:
            import json
            with open(meta) as f:
                meta = json.load(f)
        except Exception:
            return {}
    if not isinstance(meta, list):
        return {}
    return {int(e.get("proto_idx", i)): e for i, e in enumerate(meta)}


def _proto_label(meta_by_idx, comp):
    """Caption suffix with provenance: dx_type/localization (HAM) or characteristics."""
    base = comp["class_name"]
    e = meta_by_idx.get(comp["proto_global_idx"])
    if not e:
        return base
    extra = []
    if e.get("tags"):                       # HAM: [dx, dx_type, localization]
        extra = [str(t) for t in e["tags"][1:] if t]
    elif e.get("characteristics"):          # LIDC/LUNA22
        extra = [f"{k}={v}" for k, v in e["characteristics"].items()]
    return base + (" — " + ", ".join(extra) if extra else "")


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
    prototype_metadata=None,
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
    probs = torch.softmax(logits, dim=1)[0]
    # Uncertainty from the actual decision (logits), consistent with evaluation.
    uncertainty = float(compute_uncertainty_entropy(logits, temperature)[0])
    pred = int(logits.argmax(dim=1)[0].item())

    # Competing prototypes are grounded in the DECISION: the top-2 classes by
    # logit, and for each the prototype contributing most to that class's logit
    # (activation x last-layer weight). This shows the classes the model is
    # actually torn between, not arbitrary high-similarity 'magnet' prototypes.
    competitors = _competing_prototypes_by_logits(
        model, logits[0], sims[0], num_classes, K, list(class_names))
    top, second = competitors[0], competitors[1]
    meta_by_idx = _load_proto_metadata(prototype_metadata)

    disp_img = _denormalize(image_tensor)
    img_size = disp_img.shape[0]
    act_top = _prototype_activation_map(model, image_tensor,
                                        top["proto_global_idx"], img_size)
    act_2nd = _prototype_activation_map(model, image_tensor,
                                        second["proto_global_idx"], img_size)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))

    # Panel 1: input with a bounding box around each competing prototype's
    # highest-activation region. A box (rather than a colored wash) honestly
    # says "the model matched THIS region to a <class> exemplar" without
    # implying a pixel-level visual resemblance (the similarity is in the
    # network's feature space).
    axes[0].imshow(disp_img)
    for act, color in ((act_top, "red"), (act_2nd, "blue")):
        bb = _activation_bbox(act, thresh=0.6)
        if bb is not None:
            x0, y0, x1, y1 = bb
            axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        edgecolor=color, linewidth=2.5))
    axes[0].set_title("Input + matched regions\n(red box: %s, blue box: %s)"
                      % (top["class_name"], second["class_name"]), fontsize=13)
    axes[0].axis("off")

    # Panels 2 & 3: each competitor's prototype patch + activation overlay.
    for ax, comp, cmap_color, overlay in (
        (axes[1], top, "Reds", act_top),
        (axes[2], second, "Blues", act_2nd),
    ):
        label = _proto_label(meta_by_idx, comp)
        patch = _load_prototype_patch(prototype_image_dir, comp["proto_global_idx"])
        if patch is not None:
            ax.imshow(patch)
            ax.set_title("Prototype P%d [%s]\nsimilarity=%.2f"
                         % (comp["proto_global_idx"], label,
                            comp["similarity_score"]), fontsize=13)
        else:
            # No saved patch (e.g. pre-push / smoke test): show activation region.
            ax.imshow(disp_img)
            ax.imshow(overlay, cmap=cmap_color, alpha=0.5)
            ax.set_title("P%d [%s] sim=%.2f\n(patch png not found)"
                         % (comp["proto_global_idx"], label,
                            comp["similarity_score"]), fontsize=13)
        ax.axis("off")

    # Panel 4: model class-probability bar chart (the actual decision).
    prob_np = probs.cpu().numpy()
    colors = ["tab:gray"] * num_classes
    colors[top["class_id"]] = "tab:red"
    colors[second["class_id"]] = "tab:blue"
    axes[3].barh(range(num_classes), prob_np, color=colors)
    axes[3].set_yticks(range(num_classes))
    axes[3].set_yticklabels(list(class_names), fontsize=11)
    axes[3].invert_yaxis()
    axes[3].set_xlabel("Model probability", fontsize=13)
    axes[3].set_title("Predicted class distribution", fontsize=13)
    axes[3].text(0.97, 0.04, f"U = {uncertainty:.3f}",
                 transform=axes[3].transAxes, ha="right", va="bottom",
                 fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))

    true_name = (class_names[true_label] if isinstance(true_label, int)
                 and 0 <= true_label < num_classes else str(true_label))
    fig.suptitle(
        f"Uncertain Prediction — True: {true_name} | Model: {class_names[pred]} "
        f"| U = {uncertainty:.3f}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names)), 6))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=13)
    ax.set_ylabel("Uncertainty score U", fontsize=14)
    ax.set_title("Uncertainty distribution by class", fontsize=15)
    ax.tick_params(axis='y', labelsize=12)

    clear = clear_class_names if clear_class_names is not None else names
    pooled = np.concatenate(
        [np.asarray(uncertainty_scores_by_class[c], dtype=float)
         for c in clear if c in uncertainty_scores_by_class]) \
        if any(c in uncertainty_scores_by_class for c in clear) else np.array([])
    if pooled.size:
        med = float(np.median(pooled))
        med_name = "clear-class median" if clear_class_names is not None \
            else "overall median"
        ax.axhline(med, ls="--", color="gray", label=f"{med_name} = {med:.3f}")
        ax.legend(loc="upper right", fontsize=12)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(8, 7))
    if hard_labels is not None:
        hl = np.asarray(hard_labels).ravel()[mask]
        for cls in np.unique(hl):
            m = hl == cls
            name = label_names[int(cls)] if int(cls) < len(label_names) else str(cls)
            ax.scatter(s[m], u[m], alpha=0.6, s=18, label=name)
        ax.legend(title="hard label", fontsize=12, title_fontsize=12)
    else:
        ax.scatter(s, u, alpha=0.6, s=18)

    # Least-squares trend line.
    if u.size >= 2 and s.std() > 0:
        b, a = np.polyfit(s, u, 1)
        xs = np.linspace(s.min(), s.max(), 100)
        ax.plot(xs, b * xs + a, color="black", lw=1.5, ls="--")

    ax.set_xlabel("Inter-radiologist std (ground-truth aleatoric uncertainty)", fontsize=13)
    ax.set_ylabel("Model uncertainty U", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title("Model uncertainty vs. radiologist disagreement\n"
                 f"Pearson r = {corr['pearson_r']:.3f}, "
                 f"p = {corr['p_value']:.2e}  (n={corr['n_samples']})",
                 fontsize=14)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
