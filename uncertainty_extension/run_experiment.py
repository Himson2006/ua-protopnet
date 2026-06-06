"""
End-to-end UA-ProtoPNet experiment runner (Section 8).

Trains the Uncertainty-Aware ProtoPNet, calibrates temperature, evaluates on the
test set, optionally trains/evaluates the MC-Dropout and Ensemble baselines,
generates the per-sample explanation figures for the most/least uncertain test
samples, writes ``results_summary.json`` and ``prototype_metadata.json``, and
prints the paper's comparison table.

Example (Linux GPU server)::

    python -m uncertainty_extension.run_experiment \
        --dataset ham10000 --data_dir "/data/HAM Dataset" \
        --backbone resnet50 --epochs_joint 25 --lambda_u 0.5 \
        --output_dir ./results/run_001 --run_baselines

Quick Mac smoke run::

    python -m uncertainty_extension.run_experiment \
        --dataset ham10000 --data_dir "../HAM Dataset" \
        --max_samples 120 --epochs_warm 1 --epochs_joint 1 --epochs_last 1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .evaluate import calibrate_temperature, collect_predictions, run_full_evaluation
from .pp_compat import get_prototypes_per_class, resolve_device, unwrap
from .train_uncertainty_protopnet import (
    TrainConfig,
    build_ua_protopnet,
    maybe_dataparallel,
    train_ua_protopnet,
)


# --------------------------------------------------------------------------- #
# Dataset setup
# --------------------------------------------------------------------------- #

def setup_dataset(args):
    """Return (bundle, ambiguous_fn, ambiguous_label_for_eval, use_soft_labels)."""
    common = dict(num_workers=args.num_workers, pin_memory=args.pin_memory,
                  max_samples=args.max_samples)
    if args.dataset == "ham10000":
        from .datasets.ham10000 import (DEFAULT_AMBIGUOUS_PAIRS, get_ambiguous_mask,
                                        get_ham10000_dataloaders, _pairs_to_class_indices)
        bundle = get_ham10000_dataloaders(args.data_dir, **common)
        ambiguous_fn = lambda labels: get_ambiguous_mask(labels, DEFAULT_AMBIGUOUS_PAIRS)
        ambiguous_label = _pairs_to_class_indices(DEFAULT_AMBIGUOUS_PAIRS)
        return bundle, ambiguous_fn, ambiguous_label, False
    elif args.dataset == "lidc":
        import torch
        from .datasets.lidc import (AMBIGUOUS_LABEL, BINARY_AMBIGUOUS_LABEL,
                                    get_ambiguous_mask, get_lidc_dataloaders)
        if args.binary_soft:
            # Soft P(malignant) regression: the soft labels do the work, so no
            # calibration loss (ambiguous_fn marks nothing). Ambiguous eval group
            # is an is_ambiguous mask, computed per-split in main().
            bundle = get_lidc_dataloaders(args.data_dir, binary_soft=True, **common)
            ambiguous_fn = lambda labels: torch.zeros_like(labels, dtype=torch.bool)
            return bundle, ambiguous_fn, None, True
        if args.binary:
            # Held-out indeterminate nodules carry label -1; nothing in the
            # train set is ambiguous, so the calibration loss just enforces
            # certainty on the (confident) training samples. Hard labels only.
            bundle = get_lidc_dataloaders(args.data_dir, binary=True, **common)
            ambiguous_fn = lambda labels: labels == BINARY_AMBIGUOUS_LABEL
            return bundle, ambiguous_fn, bundle.ambiguous_label, False
        bundle = get_lidc_dataloaders(args.data_dir, **common)
        ambiguous_fn = get_ambiguous_mask
        return bundle, ambiguous_fn, AMBIGUOUS_LABEL, True
    raise ValueError(f"unknown dataset {args.dataset!r}")


# --------------------------------------------------------------------------- #
# Training one model (used for UA, vanilla, and ensemble members)
# --------------------------------------------------------------------------- #

def train_one_model(args, bundle, ambiguous_fn, ambiguous_label, use_soft_labels,
                    device, lambda_u, lambda_div, tag, seed):
    """Build + train a single ProtoPNet; return (model, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_ua_protopnet(
        base_architecture=args.backbone,
        num_classes=bundle.num_classes,
        prototypes_per_class=args.num_prototypes_per_class,
        pretrained=not args.no_pretrained,
    ).to(device)
    model = maybe_dataparallel(model, args.multi_gpu)
    K = get_prototypes_per_class(model)

    def eval_fn(m):
        return run_full_evaluation(
            m, bundle.val_loader, bundle.num_classes, K, device,
            ambiguous_label=ambiguous_label, temperature=args.temperature,
            uncertainty_source=args.uncertainty_source)

    cfg = TrainConfig(
        epochs_warm=args.epochs_warm, epochs_joint=args.epochs_joint,
        epochs_last=args.epochs_last, lambda_u=lambda_u, lambda_div=lambda_div,
        temperature=args.temperature, use_soft_labels=use_soft_labels,
        ckpt_dir=os.path.join(args.output_dir, "ckpt", tag),
        push_dir=os.path.join(args.output_dir, "img", tag),
        log=print)
    history = train_ua_protopnet(
        model, bundle.train_loader, bundle.val_loader, bundle.push_loader,
        device, cfg, ambiguous_fn=ambiguous_fn,
        preprocess_input_function=bundle.preprocess_input_function,
        eval_fn=eval_fn, resume=args.resume)
    return model, history, K


# --------------------------------------------------------------------------- #
# Prototype metadata (Section 10)
# --------------------------------------------------------------------------- #

def build_prototype_metadata(args, bundle, proto_source) -> List[dict]:
    """Attach human-readable tags to each prototype from its source sample."""
    meta = []
    push_ds = bundle.push_dataset
    for rec in proto_source or []:
        idx = rec.get("source_image_index", -1)
        cls = rec.get("class_id", -1)
        entry = {"proto_idx": rec.get("proto_idx"), "class_id": cls,
                 "class_name": bundle.class_names[cls] if 0 <= cls < len(bundle.class_names) else str(cls),
                 "source_image_index": idx}
        if args.dataset == "ham10000" and 0 <= idx < len(push_ds.row_meta):
            rm = push_ds.row_meta[idx]
            tags = [rm.get("dx"), rm.get("dx_type"), rm.get("localization")]
            entry["tags"] = [t for t in tags if t]
            entry["name"] = f"Prototype {rec.get('proto_idx')}: {bundle.class_names[cls]} " \
                            f"({', '.join(entry['tags'])})"
        elif args.dataset == "lidc" and 0 <= idx < len(push_ds.records):
            r = push_ds.records[idx]
            from .datasets.lidc import CHARACTERISTICS, LUNA22_CHARACTERISTICS
            # Use whichever nodule characteristics are present (pylidc vs LUNA22).
            char_keys = [c for c in (CHARACTERISTICS + LUNA22_CHARACTERISTICS)
                         if c in r]
            chars = {c: round(r[c], 2) for c in char_keys}
            entry["characteristics"] = chars
            entry["mean_malignancy"] = r.get("mean_malignancy")
            char_str = ", ".join(f"{k}={v}" for k, v in chars.items())
            entry["name"] = (f"Prototype {rec.get('proto_idx')}: {entry['class_name']} "
                             f"(malignancy={r.get('mean_malignancy', float('nan')):.1f}"
                             + (f", {char_str}" if char_str else "") + ")")
        meta.append(entry)
    return meta


# --------------------------------------------------------------------------- #
# Visualization of extreme samples (Section 8.5)
# --------------------------------------------------------------------------- #

def generate_extreme_visualizations(model, bundle, K, device, args, temperature):
    """Render explanation figures for the top-5 most & least uncertain samples."""
    from .visualize import visualize_uncertainty_explanation
    pred = collect_predictions(model, bundle.test_loader, bundle.num_classes, K,
                               device, temperature=temperature,
                               uncertainty_source=args.uncertainty_source)
    unc = pred["entropy"]
    labels = pred["labels"]
    order = np.argsort(unc)
    least = order[:5]
    most = order[-5:][::-1]

    vis_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)
    proto_dir = os.path.join(args.output_dir, "img", "ua")

    # Index into the test dataset for the chosen samples.
    test_ds = bundle.test_dataset
    for group, idxs in (("most_uncertain", most), ("least_uncertain", least)):
        for rank, gi in enumerate(idxs):
            image, label = test_ds[int(gi)][0], int(labels[int(gi)])
            save = os.path.join(vis_dir, f"{group}_{rank}_idx{int(gi)}.png")
            try:
                visualize_uncertainty_explanation(
                    model, image, label, bundle.class_names,
                    prototype_image_dir=proto_dir, save_path=save,
                    temperature=temperature, device=device)
            except Exception as e:
                print(f"[viz] skipped {group} rank {rank}: {e}")
    print(f"[viz] explanation figures written to {vis_dir}")


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #

def print_results_table(results: Dict[str, dict]):
    """Print the Section-9 comparison table to console."""
    methods = [m for m in ("vanilla", "mc_dropout", "ensemble", "ua")
               if m in results]
    headers = {"vanilla": "Vanilla", "mc_dropout": "MC Dropout",
               "ensemble": "Ensemble", "ua": "UA-ProtoPNet (Ours)"}

    def cell(m, key, sub=None, fmt="{:.4f}"):
        d = results[m]
        v = d.get(key)
        if sub and isinstance(v, dict):
            v = v.get(sub)
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "—"

    rows = [
        ("Classification Accuracy", lambda m: cell(m, "accuracy")),
        ("ECE (↓)", lambda m: cell(m, "ece")),
        ("Uncertainty AUROC (↑)", lambda m: cell(m, "uncertainty_auroc", fmt="{:.3f}")),
        ("Mean U ambiguous", lambda m: cell(m, "ambiguous_vs_clear", "u_ambiguous", "{:.3f}")),
        ("Mean U clear", lambda m: cell(m, "ambiguous_vs_clear", "u_clear", "{:.3f}")),
        ("Ambiguous/Clear ratio (↑)", lambda m: cell(m, "ambiguous_vs_clear", "ratio", "{:.2f}")),
        ("Pearson r w/ rad. std", lambda m: cell(m, "radiologist_correlation", "pearson_r", "{:.3f}")),
        ("Prototype explanation", lambda m: "Yes" if m == "ua" else "No"),
    ]
    col_w = 26
    line = "| {:<28} |".format("Metric") + "".join(
        " {:^{w}} |".format(headers[m], w=col_w) for m in methods)
    print("\n" + "=" * len(line))
    print(line)
    print("|" + "-" * (len(line) - 2) + "|")
    for name, fn in rows:
        row = "| {:<28} |".format(name) + "".join(
            " {:^{w}} |".format(fn(m), w=col_w) for m in methods)
        print(row)
    print("=" * len(line) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description="UA-ProtoPNet experiment runner")
    p.add_argument("--dataset", choices=["ham10000", "lidc"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--backbone", choices=["resnet50", "densenet121"], default="resnet50")
    p.add_argument("--num_prototypes_per_class", type=int, default=10)
    p.add_argument("--epochs_warm", type=int, default=5)
    p.add_argument("--epochs_joint", type=int, default=25)
    p.add_argument("--epochs_last", type=int, default=10)
    p.add_argument("--lambda_u", type=float, default=0.5)
    p.add_argument("--lambda_div", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--run_baselines", action="store_true")
    p.add_argument("--ensemble_size", type=int, default=3)
    p.add_argument("--output_dir", default="./results/run")
    p.add_argument("--no_pretrained", action="store_true")
    # Section 11G environment flags.
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soft_labels", choices=["auto", "on", "off"], default="auto",
                   help="Override soft-label CE. 'auto' = on for LIDC, off for HAM. "
                        "Use 'off' to train LIDC on hard labels (diagnostic).")
    p.add_argument("--binary", action="store_true",
                   help="LIDC: train benign-vs-malignant on confident nodules; "
                        "hold out the indeterminate nodules as the ambiguous test "
                        "set (recommended — the 3-class task is not learnable).")
    p.add_argument("--binary_soft", action="store_true",
                   help="LIDC: train P(malignant) to match the radiologist vote "
                        "fraction (soft regression). Confidence is calibrated to "
                        "agreement, so uncertainty should track radiologist std.")
    p.add_argument("--uncertainty_source", choices=["logits", "evidence"],
                   default="logits",
                   help="Compute uncertainty from softmax(logits) (default, "
                        "recommended) or from class evidence (per-class max "
                        "prototype similarity; saturates in practice).")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[run] dataset={args.dataset} backbone={args.backbone} device={device}")

    bundle, ambiguous_fn, ambiguous_label, use_soft = setup_dataset(args)
    if args.soft_labels == "on":
        use_soft = True
    elif args.soft_labels == "off":
        use_soft = False
    print(f"[run] train={len(bundle.train_dataset)} val={len(bundle.val_dataset)} "
          f"test={len(bundle.test_dataset)} classes={bundle.num_classes} "
          f"soft_labels={use_soft}")

    # In binary-soft mode the ambiguous (indeterminate) nodules cut across both
    # classes, so the ambiguous group is a per-split boolean mask rather than a
    # label value. Otherwise val and test share the same label-based group.
    if args.binary_soft:
        eval_ambiguous = np.array([r["is_ambiguous"]
                                   for r in bundle.val_dataset.records], dtype=bool)
        test_ambiguous = np.array([r["is_ambiguous"]
                                   for r in bundle.test_dataset.records], dtype=bool)
    else:
        eval_ambiguous = test_ambiguous = ambiguous_label

    results: Dict[str, dict] = {}

    # ---- Train UA-ProtoPNet (ours) ------------------------------------- #
    model, history, K = train_one_model(
        args, bundle, ambiguous_fn, eval_ambiguous, use_soft, device,
        args.lambda_u, args.lambda_div, tag="ua", seed=args.seed)

    # Did the model ever fit the training data? (best train acc across epochs)
    train_accs = [h.get("accuracy", 0.0) for h in history.get("train", [])]
    best_train_acc = max(train_accs) if train_accs else float("nan")
    print(f"[run] best TRAIN accuracy across epochs = {best_train_acc:.4f} "
          f"(if this is ~majority-baseline, the model is not learning the task)")

    # ---- Temperature calibration on val (Section 10) ------------------- #
    best_T = calibrate_temperature(model, bundle.val_loader, bundle.num_classes,
                                   K, device, uncertainty_source=args.uncertainty_source)
    print(f"[run] calibrated temperature T = {best_T}")

    # ---- Test evaluation ----------------------------------------------- #
    results["ua"] = run_full_evaluation(
        model, bundle.test_loader, bundle.num_classes, K, device,
        ambiguous_label=test_ambiguous, temperature=best_T,
        uncertainty_source=args.uncertainty_source)
    results["ua"]["provides_prototype_explanation"] = True
    results["ua"]["best_train_acc"] = float(best_train_acc)

    proto_meta = build_prototype_metadata(args, bundle, history.get("prototype_source"))
    with open(os.path.join(args.output_dir, "prototype_metadata.json"), "w") as f:
        json.dump(proto_meta, f, indent=2)

    generate_extreme_visualizations(model, bundle, K, device, args, best_T)

    # ---- Baselines ------------------------------------------------------ #
    if args.run_baselines:
        from .baselines import EnsembleProtoPNet, MCDropoutProtoPNet

        # Vanilla: same architecture trained WITHOUT uncertainty terms.
        vanilla, _vh, _vk = train_one_model(
            args, bundle, ambiguous_fn, eval_ambiguous, use_soft, device,
            lambda_u=0.0, lambda_div=0.0, tag="vanilla", seed=args.seed + 1)
        results["vanilla"] = run_full_evaluation(
            vanilla, bundle.test_loader, bundle.num_classes, K, device,
            ambiguous_label=test_ambiguous, temperature=best_T,
            uncertainty_source=args.uncertainty_source)

        mc = MCDropoutProtoPNet(vanilla, n_samples=30)
        results["mc_dropout"] = run_full_evaluation(
            mc, bundle.test_loader, bundle.num_classes, K, device,
            ambiguous_label=test_ambiguous, temperature=best_T,
            forward_fn=mc.forward_fn())

        members = [vanilla]
        for s in range(1, max(1, args.ensemble_size)):
            mem, _h, _k = train_one_model(
                args, bundle, ambiguous_fn, eval_ambiguous, use_soft, device,
                lambda_u=0.0, lambda_div=0.0, tag=f"ens_{s}", seed=args.seed + 10 + s)
            members.append(mem)
        ens = EnsembleProtoPNet(members)
        results["ensemble"] = run_full_evaluation(
            ens, bundle.test_loader, bundle.num_classes, K, device,
            ambiguous_label=test_ambiguous, temperature=best_T,
            forward_fn=ens.forward_fn())

    # ---- Save + print --------------------------------------------------- #
    summary = {"args": vars(args), "calibrated_temperature": best_T,
               "results": results}
    with open(os.path.join(args.output_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print_results_table(results)
    print(f"[run] results written to {args.output_dir}/results_summary.json")
    return summary


if __name__ == "__main__":
    main()
