"""
Fast end-to-end smoke test for UA-ProtoPNet (Section 11F).

Validates that the *entire* pipeline runs without error on tiny synthetic data
and a small CPU model — no real datasets, no GPU, no pretrained download. Run
this on the Mac before pushing to the Linux GPU server::

    python -m uncertainty_extension.smoke_test

Steps: (1) build a dummy 7-class HAM-style dataset on disk and load it through
the real ``ham10000`` pipeline; (2) build a minimal ProtoPNet (resnet18, 2
prototypes/class, no pretrained weights); (3) exercise the uncertainty scoring
functions; (4) one combined-loss training step; (5) a prototype push; (6) a full
evaluation step with every metric; (7) one explanation figure. Prints
``SMOKE TEST PASSED`` on success. Targets < 60 s on CPU.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image


def _make_dummy_ham(root: str, n_per_class: int = 8):
    """Write a tiny HAM10000-style dataset (images + metadata CSV) to ``root``."""
    import csv
    from .datasets.ham10000 import HAM_CLASSES

    img_dir = os.path.join(root, "HAM10000_images_part_1")
    os.makedirs(img_dir, exist_ok=True)
    rows = [("lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization")]
    k = 0
    rng = np.random.RandomState(0)
    for cls in HAM_CLASSES:
        for _ in range(n_per_class):
            image_id = f"ISIC_{k:07d}"
            arr = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(img_dir, image_id + ".jpg"))
            # two images per lesion sometimes, to exercise lesion grouping
            lesion_id = f"HAM_{k // 2:07d}"
            rows.append((lesion_id, image_id, cls, "histo", "50.0", "male", "back"))
            k += 1
    with open(os.path.join(root, "HAM10000_metadata.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


def main() -> int:
    device = torch.device("cpu")
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Dummy dataset through the real pipeline ----------------------- #
        print("[1/7] building dummy HAM10000 dataset ...")
        _make_dummy_ham(tmp, n_per_class=12)
        from .datasets.ham10000 import (DEFAULT_AMBIGUOUS_PAIRS, get_ambiguous_mask,
                                        get_ham10000_dataloaders)
        bundle = get_ham10000_dataloaders(
            tmp, train_batch_size=8, eval_batch_size=8, push_batch_size=8,
            num_workers=0)
        images, labels = next(iter(bundle.train_loader))
        assert images.shape[1:] == (3, 224, 224), images.shape

        # 2) Minimal ProtoPNet -------------------------------------------- #
        print("[2/7] building minimal ProtoPNet (resnet18, no pretrained) ...")
        from .train_uncertainty_protopnet import build_ua_protopnet
        model = build_ua_protopnet(
            base_architecture="resnet18", num_classes=bundle.num_classes,
            prototypes_per_class=2, pretrained=False).to(device)
        from .pp_compat import forward_with_similarity, get_prototypes_per_class
        K = get_prototypes_per_class(model)

        # 3) Uncertainty scoring ------------------------------------------ #
        print("[3/7] uncertainty scoring functions ...")
        from .uncertainty_scores import (compute_class_evidence,
                                         compute_prototype_competition_score,
                                         compute_uncertainty_entropy,
                                         get_competing_prototypes)
        model.eval()
        logits, _md, sims = forward_with_similarity(model, images.to(device))
        evidence = compute_class_evidence(sims, bundle.num_classes, K)
        assert evidence.shape == (images.shape[0], bundle.num_classes)
        ent = compute_uncertainty_entropy(evidence)
        comp = compute_prototype_competition_score(evidence)
        competitors = get_competing_prototypes(sims[0], bundle.num_classes, K,
                                               class_names=bundle.class_names)
        assert len(competitors) == 2
        assert (ent >= 0).all() and (ent <= 1).all()
        assert (comp >= 0).all() and (comp <= 1).all()

        # 4) One combined-loss training step ------------------------------ #
        print("[4/7] one training step with combined loss ...")
        from .train_uncertainty_protopnet import (DEFAULT_COEFS, build_optimizers,
                                                  run_epoch, set_phase)
        set_phase(model, "joint")
        opt = build_optimizers(model)["joint"]
        ambiguous_fn = lambda lb: get_ambiguous_mask(lb, DEFAULT_AMBIGUOUS_PAIRS)
        stats = run_epoch(model, bundle.train_loader, device, opt, DEFAULT_COEFS,
                          lambda_u=0.5, lambda_div=0.01, ambiguous_fn=ambiguous_fn,
                          add_uncertainty=True)
        assert np.isfinite(stats.loss), stats.loss

        # 5) Prototype push (no image saving for speed) ------------------- #
        print("[5/7] prototype push ...")
        from .push_uncertainty import push_prototypes
        proto_source = push_prototypes(
            bundle.push_loader, model, device,
            preprocess_input_function=bundle.preprocess_input_function,
            root_dir_for_saving_prototypes=None, save_images=False, log=lambda *a: None)
        assert len(proto_source) == bundle.num_classes * K

        # 6) Full evaluation with all metrics ----------------------------- #
        print("[6/7] full evaluation (ECE, AUROC, ambiguous/clear) ...")
        from .evaluate import calibrate_temperature, run_full_evaluation
        amb_idx = sorted({bundle.class_names.index(n) for pair in DEFAULT_AMBIGUOUS_PAIRS for n in pair})
        metrics = run_full_evaluation(model, bundle.test_loader, bundle.num_classes,
                                      K, device, ambiguous_label=amb_idx)
        for key in ("accuracy", "ece", "uncertainty_auroc", "ambiguous_vs_clear"):
            assert key in metrics, key
        best_T = calibrate_temperature(model, bundle.val_loader, bundle.num_classes,
                                       K, device)
        assert best_T > 0

        # 7) One explanation figure (no saved prototypes -> overlay fallback) #
        print("[7/7] explanation figure ...")
        from .visualize import (plot_uncertainty_distribution,
                                visualize_uncertainty_explanation)
        fig_path = os.path.join(tmp, "explain.png")
        visualize_uncertainty_explanation(
            model, bundle.test_dataset[0][0], int(bundle.test_dataset[0][1]),
            bundle.class_names, prototype_image_dir=None, save_path=fig_path,
            device=device)
        assert os.path.isfile(fig_path)
        # Distribution plot too.
        dist = {c: np.random.rand(10) for c in bundle.class_names}
        plot_uncertainty_distribution(dist, bundle.class_names,
                                      save_path=os.path.join(tmp, "dist.png"))

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
