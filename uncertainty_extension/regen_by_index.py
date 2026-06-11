"""
Re-render specific explanation figures BY TEST-SET INDEX, without retraining.

Both ``run_experiment`` and ``make_hero_figures`` encode the test-set index in
every figure filename (``..._idx<N>.png``). That index is stable across runs
because the data split is fixed, so the same ``idx`` always points to the same
case. This utility loads the saved checkpoint and re-renders exactly the indices
you name -- use it to regenerate your chosen hero/secondary figures with the
current (larger-font) plotting code, guaranteeing the identical case and the
identical uncertainty numbers.

Usage::

    python -m uncertainty_extension.regen_by_index \
        --output_dir ./results/ham_final \
        --indices 412 1733 \
        --out_subdir figures_regen
    # add --data_dir if the dataset path differs from the original run.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .evaluate import calibrate_temperature, collect_predictions
from .pp_compat import get_prototypes_per_class, resolve_device
from .train_uncertainty_protopnet import build_ua_protopnet, load_checkpoint
from .visualize import visualize_uncertainty_explanation
from .make_hero_figures import _load_run_args


def main(argv=None):
    p = argparse.ArgumentParser(description="Re-render figures by test-set index")
    p.add_argument("--output_dir", required=True,
                   help="Existing run dir (contains ckpt/, img/, "
                        "prototype_metadata.json, results_summary.json).")
    p.add_argument("--indices", type=int, nargs="+", required=True,
                   help="Test-set indices to render (the idx<N> in filenames).")
    p.add_argument("--data_dir", default=None,
                   help="Override the dataset path (default: from the run).")
    p.add_argument("--out_subdir", default="figures_regen",
                   help="Subfolder of output_dir to write into.")
    p.add_argument("--device", default="auto")
    cli = p.parse_args(argv)

    from .run_experiment import setup_dataset  # avoid import cycle

    a = _load_run_args(cli.output_dir, cli.data_dir)
    device = resolve_device(cli.device)
    print(f"[regen-idx] dataset={a.dataset} backbone={a.backbone} device={device}")

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

    # 3) Prototype patches + metadata + calibrated temperature (same as the run).
    proto_dir = os.path.join(cli.output_dir, "img", "ua")
    meta_path = os.path.join(cli.output_dir, "prototype_metadata.json")
    proto_meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else None
    best_T = calibrate_temperature(model, bundle.val_loader, bundle.num_classes,
                                   K, device, uncertainty_source=a.uncertainty_source)
    print(f"[regen-idx] calibrated T = {best_T}")

    # 4) Labels (for the figure title), scored exactly as in the original run.
    pred = collect_predictions(model, bundle.test_loader, bundle.num_classes, K,
                               device, temperature=best_T,
                               uncertainty_source=a.uncertainty_source)
    labels = pred["labels"]
    cls = bundle.class_names

    # 5) Render each requested index.
    out_dir = os.path.join(cli.output_dir, cli.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    test_ds = bundle.test_dataset
    n = len(test_ds)
    for gi in cli.indices:
        if not (0 <= gi < n):
            print(f"[regen-idx] skip idx {gi}: out of range (test set has {n})")
            continue
        image = test_ds[gi][0]
        label = int(labels[gi])
        tname = cls[label] if 0 <= label < len(cls) else str(label)
        fn = f"regen_idx{gi}_true-{tname}.png"
        try:
            visualize_uncertainty_explanation(
                model, image, label, cls, prototype_image_dir=proto_dir,
                save_path=os.path.join(out_dir, fn), temperature=best_T,
                device=device, prototype_metadata=proto_meta)
            print(f"[regen-idx] wrote {fn}")
        except Exception as e:
            print(f"[regen-idx] skip idx {gi}: {e}")
    print(f"[regen-idx] figures written to {out_dir}")


if __name__ == "__main__":
    main()
