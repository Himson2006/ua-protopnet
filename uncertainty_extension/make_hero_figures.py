"""
Generate additional hero / explanation figures from an ALREADY-TRAINED run,
without retraining.

It reuses the run's saved checkpoint (``<output_dir>/ckpt/ua/latest.pth``),
pushed prototype patches (``<output_dir>/img/ua/``), and prototype metadata
(``<output_dir>/prototype_metadata.json``), rebuilds the same data split, and
writes a configurable number of figures into ``<output_dir>/<out_subdir>/``.

It produces two ranked groups:
  * ``hero_*``  -- dangerous-miss candidates: a high-stakes class (melanoma /
    malignant) is a strong top-2 competitor but is NOT the prediction, ranked by
    (i) true dangerous misses and (ii) closest top1-top2 near-ties.
  * ``most_uncertain_*`` -- the highest-uncertainty cases overall.
Filenames encode true/pred/gap so you can browse without opening each file.

Usage::

    python -m uncertainty_extension.make_hero_figures \
        --output_dir ./results/ham_final \
        --n_hero 30 --n_uncertain 15
    # add --data_dir if the dataset path differs from the original run.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from .evaluate import calibrate_temperature, collect_predictions
from .pp_compat import get_prototypes_per_class, resolve_device
from .train_uncertainty_protopnet import build_ua_protopnet, load_checkpoint
from .visualize import visualize_uncertainty_explanation


def _load_run_args(output_dir: str, data_dir_override: str | None) -> SimpleNamespace:
    """Reconstruct the original run's args from results_summary.json."""
    summ_path = os.path.join(output_dir, "results_summary.json")
    if not os.path.isfile(summ_path):
        raise FileNotFoundError(
            f"{summ_path} not found -- point --output_dir at a completed run "
            f"directory (the one passed to run_experiment).")
    with open(summ_path) as f:
        saved = json.load(f)["args"]
    a = SimpleNamespace(**saved)
    if data_dir_override:
        a.data_dir = data_dir_override
    # Safe inference settings.
    a.num_workers = 0
    a.pin_memory = False
    return a


def main(argv=None):
    p = argparse.ArgumentParser(description="Make more hero/explanation figures")
    p.add_argument("--output_dir", required=True,
                   help="Existing run dir (contains ckpt/, img/, "
                        "prototype_metadata.json, results_summary.json).")
    p.add_argument("--data_dir", default=None,
                   help="Override the dataset path (default: from the run).")
    p.add_argument("--n_hero", type=int, default=20,
                   help="How many dangerous-miss hero figures to render.")
    p.add_argument("--n_uncertain", type=int, default=10,
                   help="How many most-uncertain figures to render.")
    p.add_argument("--out_subdir", default="figures_more",
                   help="Subfolder of output_dir to write into.")
    p.add_argument("--target_idx", type=int, default=None,
                   help="If set, always render this specific test-set index.")
    p.add_argument("--device", default="auto")
    cli = p.parse_args(argv)

    from .run_experiment import _high_stakes_class, setup_dataset  # avoid cycle

    a = _load_run_args(cli.output_dir, cli.data_dir)
    device = resolve_device(cli.device)
    print(f"[hero] dataset={a.dataset} backbone={a.backbone} device={device}")

    # 1) Rebuild the (identical) data split.
    bundle, _amb_fn, _amb_lab, _use_soft = setup_dataset(a)

    # 2) Build the model and load the trained weights.
    model = build_ua_protopnet(
        base_architecture=a.backbone, num_classes=bundle.num_classes,
        prototypes_per_class=a.num_prototypes_per_class, pretrained=False).to(device)
    K = get_prototypes_per_class(model)
    ckpt_dir = os.path.join(cli.output_dir, "ckpt", "ua")
    if load_checkpoint(model, {}, ckpt_dir, device) is None:
        raise FileNotFoundError(f"No checkpoint at {ckpt_dir}/latest.pth")
    model.eval()

    # 3) Prototype patches + metadata + calibrated temperature.
    proto_dir = os.path.join(cli.output_dir, "img", "ua")
    meta_path = os.path.join(cli.output_dir, "prototype_metadata.json")
    proto_meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else None
    best_T = calibrate_temperature(model, bundle.val_loader, bundle.num_classes,
                                   K, device, uncertainty_source=a.uncertainty_source)
    print(f"[hero] calibrated T = {best_T}")

    # 4) Score the test set once.
    pred = collect_predictions(model, bundle.test_loader, bundle.num_classes, K,
                               device, temperature=best_T,
                               uncertainty_source=a.uncertainty_source)
    unc, labels = pred["entropy"], pred["labels"]
    s = pred["score"]
    s = s - s.max(axis=1, keepdims=True)
    probs = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    gap = np.sort(probs, axis=1)[:, ::-1]
    gap = gap[:, 0] - gap[:, 1]
    pred_cls = probs.argmax(axis=1)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    cls = bundle.class_names

    # 5) Build the ranked groups.
    groups = {}
    hs = _high_stakes_class(a, bundle)
    if hs is not None:
        is_comp = np.array([hs in top2[i] for i in range(len(probs))])
        cand = np.where(is_comp & (pred_cls != hs))[0]
        if cand.size:
            is_miss = (labels[cand] == hs).astype(int)
            order = np.lexsort((gap[cand], -is_miss))      # miss first, tie first
            groups["hero"] = cand[order][:cli.n_hero]
            print(f"[hero] {cand.size} dangerous-miss candidates "
                  f"(high-stakes class = {cls[hs]}); rendering top {len(groups['hero'])}")
    order_u = np.argsort(unc)
    groups["most_uncertain"] = order_u[-cli.n_uncertain:][::-1]
    if cli.target_idx is not None:
        groups["target"] = np.array([cli.target_idx])

    # 6) Render.
    out_dir = os.path.join(cli.output_dir, cli.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    test_ds = bundle.test_dataset
    n_done = 0
    for group, idxs in groups.items():
        for rank, gi in enumerate(idxs):
            gi = int(gi)
            image = test_ds[gi][0]
            label = int(labels[gi])
            tname = cls[label] if 0 <= label < len(cls) else str(label)
            pname = cls[int(pred_cls[gi])]
            fn = (f"{group}_{rank:02d}_true-{tname}_pred-{pname}"
                  f"_gap{gap[gi]:.2f}_U{unc[gi]:.2f}_idx{gi}.png")
            try:
                visualize_uncertainty_explanation(
                    model, image, label, cls, prototype_image_dir=proto_dir,
                    save_path=os.path.join(out_dir, fn), temperature=best_T,
                    device=device, prototype_metadata=proto_meta)
                n_done += 1
            except Exception as e:
                print(f"[hero] skip {group} {rank} (idx {gi}): {e}")
    print(f"[hero] wrote {n_done} figures to {out_dir}")
    print("[hero] filenames encode true/pred/gap/U -- browse for the best ones; "
          "smallest 'gap' = tightest competition.")


if __name__ == "__main__":
    main()
