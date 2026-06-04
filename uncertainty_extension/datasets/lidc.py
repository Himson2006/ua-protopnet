"""
LIDC-IDRI data pipeline for UA-ProtoPNet (Section 1B).

LIDC-IDRI is a CT lung-nodule dataset in which up to four radiologists
independently rate each nodule's malignancy on a 1-5 scale. The *standard
deviation* of those ratings is a ground-truth aleatoric-uncertainty signal — the
strongest contribution of the paper is showing the model's uncertainty
correlates with it.

This module:

1. Queries every scan via ``pylidc`` and clusters annotations per nodule.
2. For each nodule with >= 3 annotators, extracts the axial slice with the
   largest consensus cross-section, crops the consensus bounding box, and
   resizes to 64x64.
3. Computes ``mean_malignancy``, ``std_malignancy`` (the aleatoric target),
   a 3-class ``hard_label`` (benign / uncertain / malignant) and a ``soft_label``
   vote distribution, plus nodule characteristics (spiculation, calcification,
   margin, lobulation, subtlety) for prototype naming.
4. Caches the extracted dataset to disk (``.npz`` + ``meta.json``).
5. Returns DataLoaders yielding
   ``(image[3,64,64], hard_label, soft_label[3], std_malignancy)``.

Class 1 ("uncertain") is the *formally defined ambiguous class* the model must
learn to be uncertain on — :func:`get_ambiguous_mask` flags it.

``pylidc`` and a configured ``~/.pylidcrc`` (pointing at the DICOM data) are
required to *build* the cache. Once cached, the loaders work without ``pylidc``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Class semantics for the 3-way structure.
LIDC_CLASSES = ["benign", "uncertain", "malignant"]
NUM_CLASSES = 3
AMBIGUOUS_LABEL = 1  # "uncertain"
PATCH_SIZE = 64
CHARACTERISTICS = ["spiculation", "calcification", "margin", "lobulation", "subtlety"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Label derivation
# --------------------------------------------------------------------------- #

def malignancy_to_hard_label(mean_malignancy: float) -> int:
    """Map mean malignancy to 0=benign, 1=uncertain, 2=malignant (spec rule)."""
    if mean_malignancy <= 2.5:
        return 0
    if mean_malignancy < 3.5:
        return 1
    return 2


def votes_to_soft_label(scores: List[float]) -> np.ndarray:
    """Convert per-radiologist 1-5 scores into a [benign, uncertain, malignant]
    vote-fraction distribution.

    Each individual score is bucketed: ``<3`` -> benign, ``==3`` -> uncertain,
    ``>3`` -> malignant. The returned vector sums to 1.
    """
    scores = np.asarray(scores, dtype=float)
    benign = float(np.mean(scores < 3))
    uncertain = float(np.mean(scores == 3))
    malignant = float(np.mean(scores > 3))
    soft = np.array([benign, uncertain, malignant], dtype=np.float32)
    total = soft.sum()
    return soft / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Extraction (requires pylidc)
# --------------------------------------------------------------------------- #

def extract_lidc_dataset(
    cache_dir: str,
    min_annotators: int = 3,
    consensus_level: float = 0.5,
    pad: int = 2,
    max_scans: Optional[int] = None,
    log=print,
) -> str:
    """Build the cached LIDC nodule-patch dataset; return the cache path.

    Skips work if the cache already exists. Requires ``pylidc`` + DICOM data.

    Parameters
    ----------
    cache_dir : str
        Directory to write ``lidc_patches.npz`` and ``lidc_meta.json``.
    min_annotators : int
        Only keep nodule clusters with at least this many annotations.
    consensus_level : float
        ``clevel`` passed to ``pylidc.utils.consensus`` (fraction of radiologists
        that must include a voxel for it to be in the consensus mask).
    pad : int
        Padding (in voxels) around the consensus bbox.
    max_scans : int, optional
        Limit number of scans processed (debugging).
    """
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, "lidc_patches.npz")
    meta_path = os.path.join(cache_dir, "lidc_meta.json")
    if os.path.isfile(npz_path) and os.path.isfile(meta_path):
        log(f"[lidc] using existing cache at {npz_path}")
        return npz_path

    try:
        import pylidc as pl
        from pylidc.utils import consensus
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "pylidc is required to build the LIDC cache. Install it and "
            "configure ~/.pylidcrc to point at the DICOM data. "
            f"(import error: {e})")

    scans = pl.query(pl.Scan).all()
    if max_scans is not None:
        scans = scans[:max_scans]
    log(f"[lidc] processing {len(scans)} scans ...")

    patches: List[np.ndarray] = []
    records: List[dict] = []

    for si, scan in enumerate(scans):
        try:
            vol = scan.to_volume()
        except Exception as e:
            log(f"[lidc] skip scan {si}: volume load failed ({e})")
            continue
        nodules = scan.cluster_annotations()
        for nodule_anns in nodules:
            if len(nodule_anns) < min_annotators:
                continue
            mal_scores = [a.malignancy for a in nodule_anns]
            mean_mal = float(np.mean(mal_scores))
            std_mal = float(np.std(mal_scores))

            try:
                cmask, cbbox, _ = consensus(nodule_anns, clevel=consensus_level,
                                            pad=pad)
            except Exception as e:
                log(f"[lidc] skip nodule (consensus failed: {e})")
                continue

            # Axial slice (last axis = z) with the largest consensus area.
            areas = cmask.sum(axis=(0, 1))
            if areas.max() == 0:
                continue
            z = int(np.argmax(areas))
            sub_vol = vol[cbbox]                      # cropped CT region
            patch2d = sub_vol[:, :, z].astype(np.float32)

            # Normalize HU to [0,1] using a lung window for stable contrast.
            patch2d = _window_normalize(patch2d)
            patch2d = _resize_2d(patch2d, PATCH_SIZE)

            chars = {c: float(np.mean([getattr(a, c) for a in nodule_anns]))
                     for c in CHARACTERISTICS}

            patches.append(patch2d)
            records.append({
                "scan_index": si,
                "patient_id": getattr(scan, "patient_id", str(si)),
                "n_annotators": len(nodule_anns),
                "mean_malignancy": mean_mal,
                "std_malignancy": std_mal,
                "hard_label": malignancy_to_hard_label(mean_mal),
                "soft_label": votes_to_soft_label(mal_scores).tolist(),
                "malignancy_scores": [float(s) for s in mal_scores],
                **chars,
            })

    if not patches:
        raise RuntimeError("[lidc] no nodules extracted — check pylidc config.")

    np.savez_compressed(npz_path, patches=np.stack(patches))
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    log(f"[lidc] cached {len(patches)} nodule patches -> {npz_path}")
    return npz_path


def _window_normalize(patch: np.ndarray, center: float = -600.0,
                      width: float = 1500.0) -> np.ndarray:
    """Apply a CT lung window and scale to [0,1]."""
    lo, hi = center - width / 2, center + width / 2
    patch = np.clip(patch, lo, hi)
    return (patch - lo) / (hi - lo + 1e-8)


def _resize_2d(patch: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D float patch to size x size (cv2 if available, else torch)."""
    try:
        import cv2
        return cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
    except Exception:
        t = torch.tensor(patch).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bicubic",
                                             align_corners=False)
        return t[0, 0].numpy()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class LIDCDataset(Dataset):
    """LIDC nodule patches loaded from the cache.

    Each item: ``(image[3,64,64] float, hard_label int, soft_label[3] float,
    std_malignancy float)``. The single-channel CT patch is replicated to 3
    channels and ImageNet-normalized so it can feed an ImageNet-pretrained
    backbone, exactly as Section 10 specifies.
    """

    def __init__(self, patches: np.ndarray, records: List[dict],
                 train: bool = False, normalize: bool = True):
        self.patches = patches
        self.records = records
        self.train = train
        self.normalize = normalize
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        patch = torch.tensor(self.patches[idx], dtype=torch.float32)  # [64,64] in [0,1]
        if self.train:
            if torch.rand(()) < 0.5:
                patch = torch.flip(patch, dims=[1])
            if torch.rand(()) < 0.5:
                patch = torch.flip(patch, dims=[0])
        image = patch.unsqueeze(0).repeat(3, 1, 1)   # grayscale -> 3 channels
        if self.normalize:
            image = (image - self._mean) / self._std
        rec = self.records[idx]
        hard = int(rec["hard_label"])
        soft = torch.tensor(rec["soft_label"], dtype=torch.float32)
        std = float(rec["std_malignancy"])
        return image, hard, soft, std

    def get_all_labels(self) -> torch.Tensor:
        return torch.tensor([r["hard_label"] for r in self.records], dtype=torch.long)

    def get_all_std(self) -> np.ndarray:
        return np.array([r["std_malignancy"] for r in self.records], dtype=float)


# --------------------------------------------------------------------------- #
# Split + loaders
# --------------------------------------------------------------------------- #

def imagenet_preprocess(batch: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize a batch of [0,1] 3-channel patches (push preprocess fn)."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _stratified_split(records, seed, ratios):
    rng = np.random.RandomState(seed)
    by_label: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
    for i, r in enumerate(records):
        by_label[int(r["hard_label"])].append(i)
    train, val, test = [], [], []
    for cls, idxs in by_label.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        train += idxs[:n_tr].tolist()
        val += idxs[n_tr:n_tr + n_va].tolist()
        test += idxs[n_tr + n_va:].tolist()
    return train, val, test


@dataclass
class LIDCBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    push_loader: DataLoader
    train_dataset: LIDCDataset
    val_dataset: LIDCDataset
    test_dataset: LIDCDataset
    push_dataset: LIDCDataset
    preprocess_input_function: object = staticmethod(imagenet_preprocess)
    class_names: List[str] = field(default_factory=lambda: list(LIDC_CLASSES))
    num_classes: int = NUM_CLASSES
    ambiguous_label: int = AMBIGUOUS_LABEL


def get_lidc_dataloaders(
    data_dir: str,
    train_batch_size: int = 64,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    max_samples: Optional[int] = None,
    build_if_missing: bool = True,
    log=print,
) -> LIDCBundle:
    """Build train/val/test loaders for the cached LIDC dataset.

    ``data_dir`` is the cache directory (``lidc_patches.npz`` + ``lidc_meta.json``).
    If the cache is missing and ``build_if_missing``, :func:`extract_lidc_dataset`
    is invoked (needs ``pylidc``).
    """
    npz_path = os.path.join(data_dir, "lidc_patches.npz")
    meta_path = os.path.join(data_dir, "lidc_meta.json")
    if not (os.path.isfile(npz_path) and os.path.isfile(meta_path)):
        if build_if_missing:
            extract_lidc_dataset(data_dir, log=log)
        else:
            raise FileNotFoundError(f"LIDC cache not found in {data_dir}")

    patches = np.load(npz_path)["patches"]
    with open(meta_path) as f:
        records = json.load(f)

    if max_samples is not None and len(records) > max_samples:
        idx = np.random.RandomState(seed).choice(len(records), max_samples,
                                                  replace=False)
        patches = patches[idx]
        records = [records[i] for i in idx]

    tr, va, te = _stratified_split(records, seed, ratios)

    def subset(indices, train):
        return LIDCDataset(patches[indices],
                           [records[i] for i in indices], train=train)

    train_ds = subset(np.array(tr), True)
    val_ds = subset(np.array(va), False)
    test_ds = subset(np.array(te), False)
    # Push set: train images, UNnormalized [0,1] 3-channel, for prototype saving.
    push_ds = LIDCDataset(patches[np.array(tr)], [records[i] for i in tr],
                          train=False, normalize=False)

    g = torch.Generator(); g.manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    return LIDCBundle(
        train_loader=DataLoader(train_ds, batch_size=train_batch_size,
                                shuffle=True, generator=g, **common),
        val_loader=DataLoader(val_ds, batch_size=eval_batch_size,
                              shuffle=False, **common),
        test_loader=DataLoader(test_ds, batch_size=eval_batch_size,
                               shuffle=False, **common),
        push_loader=DataLoader(push_ds, batch_size=eval_batch_size,
                               shuffle=False, **common),
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        push_dataset=push_ds,
    )


def get_ambiguous_mask(labels: torch.Tensor) -> torch.Tensor:
    """Mark the formally-defined ambiguous class (label == 1) as True."""
    labels = torch.as_tensor(labels)
    return labels == AMBIGUOUS_LABEL


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build/inspect the LIDC cache")
    p.add_argument("--data_dir", required=True, help="LIDC cache directory")
    p.add_argument("--build", action="store_true", help="Force extraction")
    args = p.parse_args()
    if args.build:
        extract_lidc_dataset(args.data_dir)
    bundle = get_lidc_dataloaders(args.data_dir, build_if_missing=args.build)
    for name, ds in (("train", bundle.train_dataset), ("val", bundle.val_dataset),
                     ("test", bundle.test_dataset)):
        labels = ds.get_all_labels()
        counts = torch.bincount(labels, minlength=NUM_CLASSES).tolist()
        print(f"  {name}: {len(ds)} nodules  dist={dict(zip(LIDC_CLASSES, counts))}")
