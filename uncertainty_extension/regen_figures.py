"""
Regenerate the summary figures (correlation scatter, uncertainty distribution)
from a SAVED CHECKPOINT, without retraining.

``run_experiment`` writes a resumable checkpoint to ``<output_dir>/ckpt/ua/
latest.pth`` each epoch. This utility rebuilds the model, loads that checkpoint,
runs inference over the test set, and re-emits the figures with the current
(fixed) plotting code. Useful when only a figure changed (e.g. the scatter
coloring fix) and a full multi-hour rerun is unnecessary.

Example (regenerate the corrected LIDC scatter)::

    python -m uncertainty_extension.regen_figures \
        --dataset lidc --data_dir ~/luna22/luna22_cache_crop64 --binary \
        --ckpt ./results/luna_Final/ckpt/ua/latest.pth \
        --backbone resnet50 --num_prototypes_per_class 10 \
        --num_workers 4 --output_dir ./results/luna_scatter
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch

from .evaluate import calibrate_temperature, collect_predictions
from .pp_compat import get_prototypes_per_class, resolve_device
from .train_uncertainty_protopnet import build_ua_protopnet
from .visualize import plot_correlation_scatter, plot_uncertainty_distribution


def build_parser():
    p = argparse.ArgumentParser(description="Regenerate figures from a checkpoint")
    p.add_argument("--dataset", choices=["ham10000", "lidc"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True, help="Path to latest.pth checkpoint")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--num_prototypes_per_class", type=int, default=10)
    p.add_argument("--binary", action="store_true")
    p.add_argument("--binary_soft", action="store_true")
    p.add_argument("--uncertainty_source", choices=["logits", "evidence", "distance"],
                   default="logits")
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_calibrate", action="store_true",
                   help="Skip val temperature calibration; use T=1.")
    return p


def _build_bundle(args):
    common = dict(num_workers=args.num_workers)
    if args.dataset == "lidc":
        from .datasets.lidc import get_lidc_dataloaders
        return get_lidc_dataloaders(args.data_dir, binary=args.binary,
                                    binary_soft=args.binary_soft, **common)
    from .datasets.ham10000 import get_ham10000_dataloaders
    return get_ham10000_dataloaders(args.data_dir, **common)


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)

    bundle = _build_bundle(args)
    model = build_ua_protopnet(
        args.backbone, num_classes=bundle.num_classes,
        prototypes_per_class=args.num_prototypes_per_class, pretrained=False).to(device)

    state = torch.load(args.ckpt, map_location=device)
    state_dict = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()
    K = get_prototypes_per_class(model)
    print(f"[regen] loaded checkpoint {args.ckpt} "
          f"(num_classes={bundle.num_classes}, K={K})")

    best_T = 1.0 if args.no_calibrate else calibrate_temperature(
        model, bundle.val_loader, bundle.num_classes, K, device,
        uncertainty_source=args.uncertainty_source)
    pred = collect_predictions(model, bundle.test_loader, bundle.num_classes, K,
                               device, temperature=best_T,
                               uncertainty_source=args.uncertainty_source)
    unc, labels = pred["entropy"], pred["labels"]

    # Per-class uncertainty distribution.
    by_class = {bundle.class_names[c]: unc[labels == c]
                for c in range(bundle.num_classes) if (labels == c).any()}
    if (labels < 0).any():
        by_class["indeterminate"] = unc[labels < 0]
    plot_uncertainty_distribution(
        by_class, list(by_class.keys()),
        save_path=os.path.join(vis_dir, "uncertainty_distribution.png"))

    # Correlation scatter (LIDC only), colored by the true 3-class ground truth.
    if "radiologist_std" in pred:
        recs = getattr(bundle.test_dataset, "records", None)
        if recs is not None and len(recs) == len(unc):
            from .datasets.lidc import LIDC_CLASSES
            scatter_labels = np.array([int(r.get("hard_label", 0)) for r in recs])
            scatter_names = LIDC_CLASSES
        else:
            scatter_labels = np.clip(labels, 0, None)
            scatter_names = bundle.class_names
        plot_correlation_scatter(
            unc, pred["radiologist_std"],
            save_path=os.path.join(vis_dir, "correlation_scatter.png"),
            hard_labels=scatter_labels, label_names=scatter_names)

    print(f"[regen] figures written to {vis_dir} (T={best_T})")


if __name__ == "__main__":
    main()
