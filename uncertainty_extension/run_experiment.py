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

def _high_stakes_class(args, bundle):
    """Index of the class whose missed-detection is most dangerous (for hero cases)."""
    if args.dataset == "ham10000" and "mel" in bundle.class_names:
        return bundle.class_names.index("mel")          # melanoma
    if args.dataset == "lidc":
        return bundle.num_classes - 1                   # malignant (last class)
    return None


def generate_extreme_visualizations(model, bundle, K, device, args, temperature,
                                    prototype_metadata=None):
    """Render explanation figures: most/least uncertain + dangerous-miss 'hero' cases.

    Hero cases = samples where a high-stakes class (e.g. melanoma) is a strong
    competitor in the top-2 but is NOT the prediction — exactly the situation the
    competing-prototype explanation is meant to surface to a clinician.
    """
    from .visualize import visualize_uncertainty_explanation
    pred = collect_predictions(model, bundle.test_loader, bundle.num_classes, K,
                               device, temperature=temperature,
                               uncertainty_source=args.uncertainty_source)
    unc = pred["entropy"]
    labels = pred["labels"]
    order = np.argsort(unc)
    groups = {"most_uncertain": order[-5:][::-1], "least_uncertain": order[:5]}

    # Hero (dangerous-miss) selection: cases where the high-stakes class (e.g.
    # melanoma) is a strong top-2 competitor but is NOT the prediction, i.e. the
    # model is leaning the *wrong* (benign) way. We surface the strongest such
    # cases, prioritizing (i) genuine NEAR-TIES (small top1-top2 probability gap,
    # where the competing-prototype explanation is most compelling) and (ii) true
    # dangerous misses (ground-truth label == the high-stakes class).
    hs = _high_stakes_class(args, bundle)
    if hs is not None and "score" in pred:
        s = pred["score"]
        s = s - s.max(axis=1, keepdims=True)
        probs = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        gap = sorted_p[:, 0] - sorted_p[:, 1]          # 0 = perfect tie
        pred_cls = probs.argmax(axis=1)
        top2 = np.argsort(-probs, axis=1)[:, :2]
        is_competitor = np.array([hs in top2[i] for i in range(len(probs))])
        cand = np.where(is_competitor & (pred_cls != hs))[0]
        if cand.size:
            is_miss = (labels[cand] == hs).astype(int)   # true high-stakes case
            # Primary: dangerous misses first; secondary: closest tie first.
            order_c = np.lexsort((gap[cand], -is_miss))
            groups["hero"] = cand[order_c][:5]
            h0 = int(groups["hero"][0])
            tl = labels[h0]
            tname = (bundle.class_names[tl] if 0 <= tl < bundle.num_classes else str(tl))
            print(f"[viz] best hero (hero_0): true={tname}, "
                  f"pred={bundle.class_names[pred_cls[h0]]}, "
                  f"top1-top2 gap={gap[h0]:.3f} (smaller = tighter competition)")

    vis_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)
    proto_dir = os.path.join(args.output_dir, "img", "ua")
    test_ds = bundle.test_dataset
    for group, idxs in groups.items():
        for rank, gi in enumerate(idxs):
            image, label = test_ds[int(gi)][0], int(labels[int(gi)])
            save = os.path.join(vis_dir, f"{group}_{rank}_idx{int(gi)}.png")
            try:
                visualize_uncertainty_explanation(
                    model, image, label, bundle.class_names,
                    prototype_image_dir=proto_dir, save_path=save,
                    temperature=temperature, device=device,
                    prototype_metadata=prototype_metadata)
            except Exception as e:
                print(f"[viz] skipped {group} rank {rank}: {e}")
    print(f"[viz] explanation figures written to {vis_dir}"
          + (f" (incl. {len(groups.get('hero', []))} dangerous-miss hero cases)"
             if "hero" in groups else ""))

    # Summary plots: per-class uncertainty distribution, and (LIDC) the
    # uncertainty-vs-radiologist-std scatter for the negative-result section.
    from .visualize import plot_correlation_scatter, plot_uncertainty_distribution
    try:
        by_class = {bundle.class_names[c]: unc[labels == c]
                    for c in range(bundle.num_classes) if (labels == c).any()}
        if (labels < 0).any():
            by_class["indeterminate"] = unc[labels < 0]
        plot_uncertainty_distribution(
            by_class, list(by_class.keys()),
            save_path=os.path.join(vis_dir, "uncertainty_distribution.png"))
    except Exception as e:
        print(f"[viz] skipped distribution plot: {e}")
    if "radiologist_std" in pred:
        try:
            # Color by the ORIGINAL 3-class ground truth (benign/uncertain/
            # malignant), not the binary training label, so the indeterminate
            # nodules are visible. test_loader is unshuffled, so record order
            # aligns with the collected predictions.
            recs = getattr(bundle.test_dataset, "records", None)
            if recs is not None and len(recs) == len(unc):
                from .datasets.lidc import LIDC_CLASSES
                scatter_labels = np.array([int(r.get("hard_label", 0))
                                           for r in recs])
                scatter_names = LIDC_CLASSES
            else:
                scatter_labels = np.clip(labels, 0, None)
                scatter_names = bundle.class_names
            plot_correlation_scatter(
                unc, pred["radiologist_std"],
                save_path=os.path.join(vis_dir, "correlation_scatter.png"),
                hard_labels=scatter_labels, label_names=scatter_names)
        except Exception as e:
            print(f"[viz] skipped correlation scatter: {e}")


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
    p.add_argument("--uncertainty_source", choices=["logits", "evidence", "distance"],
                   default="logits",
                   help="Uncertainty signal: 'logits' = entropy of softmax(logits) "
                        "(default); 'evidence' = entropy of class evidence "
                        "(saturates); 'distance' = atypicality (distance to nearest "
                        "prototype).")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated model seeds for multi-seed runs, e.g. "
                        "'0,1,2'. Reports mean+/-std per metric. The data split is "
                        "fixed by --seed; only model init/training varies. Default: "
                        "single run with --seed.")
    return p


def _difficulty_array(args, dataset):
    """Image-independent case-difficulty score per sample (LIDC only).

    Returns boundary-proximity = -|mean_malignancy - 3| so that a POSITIVE
    correlation with uncertainty means 'more uncertain near the benign/malignant
    boundary'. The model never sees this, so the correlation is not circular.
    """
    if args.dataset != "lidc":
        return None
    return np.array([-(abs(float(r.get("mean_malignancy", 3.0)) - 3.0))
                     for r in dataset.records], dtype=float)


def single_run(args, bundle, ambiguous_fn, eval_ambiguous, test_ambiguous,
               use_soft, test_difficulty, device, seed, do_artifacts):
    """Train + evaluate UA (and baselines) once at the given model seed."""
    results: Dict[str, dict] = {}
    usrc = args.uncertainty_source

    model, history, K = train_one_model(
        args, bundle, ambiguous_fn, eval_ambiguous, use_soft, device,
        args.lambda_u, args.lambda_div, tag="ua", seed=seed)

    train_accs = [h.get("accuracy", 0.0) for h in history.get("train", [])]
    best_train_acc = max(train_accs) if train_accs else float("nan")
    print(f"[seed {seed}] best TRAIN accuracy across epochs = {best_train_acc:.4f}")

    best_T = calibrate_temperature(model, bundle.val_loader, bundle.num_classes,
                                   K, device, uncertainty_source=usrc)
    print(f"[seed {seed}] calibrated temperature T = {best_T}")

    results["ua"] = run_full_evaluation(
        model, bundle.test_loader, bundle.num_classes, K, device,
        ambiguous_label=test_ambiguous, temperature=best_T,
        uncertainty_source=usrc, difficulty=test_difficulty)
    results["ua"]["provides_prototype_explanation"] = True
    results["ua"]["best_train_acc"] = float(best_train_acc)

    if do_artifacts:
        proto_meta = build_prototype_metadata(args, bundle,
                                              history.get("prototype_source"))
        with open(os.path.join(args.output_dir, "prototype_metadata.json"), "w") as f:
            json.dump(proto_meta, f, indent=2)
        generate_extreme_visualizations(model, bundle, K, device, args, best_T,
                                        prototype_metadata=proto_meta)

    if args.run_baselines:
        from .baselines import EnsembleProtoPNet, MCDropoutProtoPNet
        vanilla, _vh, _vk = train_one_model(
            args, bundle, ambiguous_fn, eval_ambiguous, use_soft, device,
            lambda_u=0.0, lambda_div=0.0, tag="vanilla", seed=seed + 1)
        results["vanilla"] = run_full_evaluation(
            vanilla, bundle.test_loader, bundle.num_classes, K, device,
            ambiguous_label=test_ambiguous, temperature=best_T,
            uncertainty_source=usrc, difficulty=test_difficulty)

        mc = MCDropoutProtoPNet(vanilla, n_samples=30)
        results["mc_dropout"] = run_full_evaluation(
            mc, bundle.test_loader, bundle.num_classes, K, device,
            ambiguous_label=test_ambiguous, temperature=best_T,
            forward_fn=mc.forward_fn(), difficulty=test_difficulty)

        members = [vanilla]
        for s in range(1, max(1, args.ensemble_size)):
            mem, _h, _k = train_one_model(
                args, bundle, ambiguous_fn, eval_ambiguous, use_soft, device,
                lambda_u=0.0, lambda_div=0.0, tag=f"ens_{s}", seed=seed + 10 + s)
            members.append(mem)
        ens = EnsembleProtoPNet(members)
        results["ensemble"] = run_full_evaluation(
            ens, bundle.test_loader, bundle.num_classes, K, device,
            ambiguous_label=test_ambiguous, temperature=best_T,
            forward_fn=ens.forward_fn(), difficulty=test_difficulty)

    return results, best_T


#: (json path, display label, format) for metrics aggregated across seeds.
_AGG_METRICS = [
    ("accuracy", "Accuracy", "{:.3f}"),
    ("ece", "ECE", "{:.3f}"),
    ("uncertainty_auroc", "Unc. AUROC", "{:.3f}"),
    ("ambiguous_vs_clear.ratio", "Ambig/Clear ratio", "{:.2f}"),
    ("ambiguous_vs_clear.p_value", "  (ratio p-value)", "{:.3f}"),
    ("radiologist_correlation.pearson_r", "Pearson r (rad std)", "{:.3f}"),
    ("difficulty_correlation.pearson_r", "Pearson r (difficulty)", "{:.3f}"),
]


def _get_path(d, path):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


def aggregate_over_seeds(results_per_seed):
    """Aggregate per-seed results into {method: {path: (mean, std, n)}}."""
    methods = []
    for r in results_per_seed:
        for m in r:
            if m not in methods:
                methods.append(m)
    agg = {}
    for m in methods:
        agg[m] = {}
        for path, _label, _fmt in _AGG_METRICS:
            vals = [_get_path(r[m], path) for r in results_per_seed if m in r]
            vals = [v for v in vals if v is not None and v == v]
            if vals:
                agg[m][path] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return agg


def print_aggregated_table(agg, n_seeds):
    methods = [m for m in ("vanilla", "mc_dropout", "ensemble", "ua") if m in agg]
    headers = {"vanilla": "Vanilla", "mc_dropout": "MC Dropout",
               "ensemble": "Ensemble", "ua": "UA-ProtoPNet (Ours)"}
    cw = 24
    line = "| {:<24} |".format(f"Metric (mean+/-std, n={n_seeds})") + "".join(
        " {:^{w}} |".format(headers[m], w=cw) for m in methods)
    print("\n" + "=" * len(line)); print(line)
    print("|" + "-" * (len(line) - 2) + "|")
    for path, label, fmt in _AGG_METRICS:
        cells = []
        for m in methods:
            v = agg[m].get(path)
            cells.append(f"{fmt.format(v[0])}+/-{fmt.format(v[1])}" if v else "—")
        print("| {:<24} |".format(label) + "".join(
            " {:^{w}} |".format(c, w=cw) for c in cells))
    # Interpretability row.
    print("| {:<24} |".format("Prototype explanation") + "".join(
        " {:^{w}} |".format("Yes" if m == "ua" else "No", w=cw) for m in methods))
    print("=" * len(line) + "\n")


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

    # Ambiguous group: per-split mask in binary-soft mode (indeterminate nodules
    # cut across both classes), else a shared label value.
    if args.binary_soft:
        eval_ambiguous = np.array([r["is_ambiguous"]
                                   for r in bundle.val_dataset.records], dtype=bool)
        test_ambiguous = np.array([r["is_ambiguous"]
                                   for r in bundle.test_dataset.records], dtype=bool)
    else:
        eval_ambiguous = test_ambiguous = ambiguous_label
    test_difficulty = _difficulty_array(args, bundle.test_dataset)

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])

    results_per_seed = []
    for i, seed in enumerate(seeds):
        print(f"\n===== seed {seed} ({i + 1}/{len(seeds)}) =====")
        res, best_T = single_run(args, bundle, ambiguous_fn, eval_ambiguous,
                                 test_ambiguous, use_soft, test_difficulty,
                                 device, seed, do_artifacts=(i == 0))
        results_per_seed.append(res)

    summary = {"args": vars(args), "seeds": seeds,
               "results_per_seed": results_per_seed}
    if len(seeds) > 1:
        agg = aggregate_over_seeds(results_per_seed)
        summary["aggregated"] = {m: {p: list(v) for p, v in d.items()}
                                 for m, d in agg.items()}
        print_aggregated_table(agg, len(seeds))
    else:
        print_results_table(results_per_seed[0])

    with open(os.path.join(args.output_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[run] results written to {args.output_dir}/results_summary.json")
    return summary


if __name__ == "__main__":
    main()
