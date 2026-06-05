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
# LUNA22-ISMI extractor (preprocessed nodule patches — no pylidc/DICOM needed)
# --------------------------------------------------------------------------- #

#: Characteristics available in the LUNA22-ISMI metadata (per-radiologist lists).
LUNA22_CHARACTERISTICS = ["texture", "calcification", "diameter"]


def _luna22_hard_label(mean_malignancy: float) -> int:
    """LUNA22 label rule: 0 benign (<=2.5), 1 uncertain (2.5<m<=3.5), 2 malignant."""
    if mean_malignancy <= 2.5:
        return 0
    if mean_malignancy <= 3.5:
        return 1
    return 2


def _load_nii_slice(path: str, slice_axis: int = 2) -> np.ndarray:
    """Load a .nii.gz patch and return its centered 2D slice along ``slice_axis``."""
    try:
        import nibabel as nib
        vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    except Exception:
        import SimpleITK as sitk
        # SimpleITK returns (z, y, x); transpose to (x, y, z) to match the spec.
        vol = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(path)), (2, 1, 0))
        vol = vol.astype(np.float32)
    center = vol.shape[slice_axis] // 2
    return np.take(vol, center, axis=slice_axis)


def extract_luna22_dataset(
    luna_dir: str,
    cache_dir: str,
    patch_size: int = 224,
    slice_axis: int = 2,
    hu_window: bool = True,
    log=print,
) -> str:
    """Build the standard LIDC cache from LUNA22-ISMI preprocessed patches.

    Produces ``lidc_patches.npz`` + ``lidc_meta.json`` in ``cache_dir`` so the
    existing :func:`get_lidc_dataloaders` / ``run_experiment --dataset lidc``
    work unchanged. No pylidc / DICOM required.

    Parameters
    ----------
    luna_dir : str
        Folder containing ``LIDC-IDRI_1176.npy`` and the nodule ``.nii.gz`` files
        (either already unzipped, or ``LIDC-IDRI_1176.zip`` which is unzipped
        into ``<luna_dir>/images`` on first run).
    cache_dir : str
        Output directory for the cache (this is the ``--data_dir`` you later pass
        to ``run_experiment``).
    patch_size : int
        Output square size of the 2D nodule slice (224 to match the backbone).
    slice_axis : int
        Axis of the centered slice (z=2 for the (x,y,z) patches; nodule centered
        at index 32).
    hu_window : bool
        Apply a CT lung window (HU center -600, width 1500) -> [0,1]. Disable if
        your patches are already intensity-normalized.
    """
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, "lidc_patches.npz")
    meta_path = os.path.join(cache_dir, "lidc_meta.json")
    if os.path.isfile(npz_path) and os.path.isfile(meta_path):
        log(f"[luna22] using existing cache at {npz_path}")
        return npz_path

    npy_path = os.path.join(luna_dir, "LIDC-IDRI_1176.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(
            f"LIDC-IDRI_1176.npy not found in {luna_dir}. Point --luna_dir at the "
            f"folder containing the .npy and the nodule .nii.gz files / zip.")
    entries = np.load(npy_path, allow_pickle=True)
    log(f"[luna22] loaded metadata for {len(entries)} nodules")

    # Locate the image files (unzip on first run if needed), index by filename.
    img_index = _build_luna22_image_index(luna_dir, log=log)

    patches: List[np.ndarray] = []
    records: List[dict] = []
    n_missing = 0
    for nod in entries:
        fname = nod["Filename"]
        path = img_index.get(fname) or img_index.get(os.path.basename(fname))
        if path is None:
            n_missing += 1
            continue
        try:
            sl = _load_nii_slice(path, slice_axis=slice_axis)
        except Exception as e:
            log(f"[luna22] skip {fname}: load failed ({e})")
            continue
        if hu_window:
            sl = _window_normalize(sl)
        else:
            mn, mx = float(sl.min()), float(sl.max())
            sl = (sl - mn) / (mx - mn + 1e-8)
        sl = _resize_2d(sl.astype(np.float32), patch_size)

        mal = [float(m) for m in nod["Malignancy"]]
        mean_mal = float(np.mean(mal))
        rec = {
            "series_uid": str(nod.get("SeriesInstanceUID", "")),
            "filename": str(fname),
            "n_annotators": len(mal),
            "mean_malignancy": mean_mal,
            "std_malignancy": float(np.std(mal)),
            "hard_label": _luna22_hard_label(mean_mal),
            "soft_label": votes_to_soft_label(mal).tolist(),
            "malignancy_scores": mal,
        }
        for key, src in (("texture", "Texture"), ("calcification", "Calcification"),
                         ("diameter", "Diameter")):
            if src in nod and nod[src] is not None and len(nod[src]) > 0:
                rec[key] = float(np.mean([float(v) for v in nod[src]]))
        patches.append(sl)
        records.append(rec)

    if not patches:
        raise RuntimeError("[luna22] no patches extracted — check image files.")
    if n_missing:
        log(f"[luna22] WARNING: {n_missing} nodules had no matching image file")

    np.savez_compressed(npz_path, patches=np.stack(patches))
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    dist = {LIDC_CLASSES[c]: sum(r["hard_label"] == c for r in records)
            for c in range(NUM_CLASSES)}
    log(f"[luna22] cached {len(patches)} patches -> {npz_path}  class dist={dist}")
    return npz_path


def _build_luna22_image_index(luna_dir: str, log=print) -> Dict[str, str]:
    """Map nodule filename -> path, unzipping LIDC-IDRI_1176.zip if necessary."""
    import zipfile

    def _index(root: str) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        for r, _d, files in os.walk(root):
            for fn in files:
                if fn.endswith(".nii.gz") or fn.endswith(".nii"):
                    idx.setdefault(fn, os.path.join(r, fn))
        return idx

    idx = _index(luna_dir)
    if idx:
        return idx
    zip_path = os.path.join(luna_dir, "LIDC-IDRI_1176.zip")
    if os.path.isfile(zip_path):
        out = os.path.join(luna_dir, "images")
        log(f"[luna22] unzipping {zip_path} -> {out} (first run only)")
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        return _index(out)
    raise FileNotFoundError(
        f"No .nii.gz images and no LIDC-IDRI_1176.zip found in {luna_dir}")


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
                 train: bool = False, normalize: bool = True,
                 label_key: str = "hard_label"):
        self.patches = patches
        self.records = records
        self.train = train
        self.normalize = normalize
        # Which record field supplies the integer class label. Binary mode uses
        # 'binary_label' (0=benign, 1=malignant, -1=held-out ambiguous).
        self.label_key = label_key
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
        hard = int(rec.get(self.label_key, rec["hard_label"]))
        soft = torch.tensor(rec["soft_label"], dtype=torch.float32)
        std = float(rec["std_malignancy"])
        return image, hard, soft, std

    def get_all_labels(self) -> torch.Tensor:
        return torch.tensor([r.get(self.label_key, r["hard_label"])
                             for r in self.records], dtype=torch.long)

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


#: Class names for the binary (benign vs malignant) reformulation.
BINARY_CLASSES = ["benign", "malignant"]
#: Sentinel label for held-out ambiguous (indeterminate) nodules in binary mode.
BINARY_AMBIGUOUS_LABEL = -1


def _binary_split(records, seed, ratios):
    """Split for binary mode: stratify the CONFIDENT nodules (binary_label 0/1)
    into train/val/test; route ALL ambiguous (label -1) nodules into test."""
    rng = np.random.RandomState(seed)
    by = {0: [], 1: []}
    ambiguous = []
    for i, r in enumerate(records):
        bl = r["binary_label"]
        (by[bl] if bl in (0, 1) else ambiguous).append(i)
    train, val, test = [], [], []
    for cls, idxs in by.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        train += idxs[:n_tr].tolist()
        val += idxs[n_tr:n_tr + n_va].tolist()
        test += idxs[n_tr + n_va:].tolist()
    test += ambiguous  # indeterminate nodules are a held-out test-only set
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
    binary: bool = False


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
    binary: bool = False,
    log=print,
) -> LIDCBundle:
    """Build train/val/test loaders for the cached LIDC dataset.

    ``data_dir`` is the cache directory (``lidc_patches.npz`` + ``lidc_meta.json``).
    If the cache is missing and ``build_if_missing``, :func:`extract_lidc_dataset`
    is invoked (needs ``pylidc``).

    binary : bool
        If True, reformulate as **benign (0) vs malignant (2 -> 1)** trained on
        the confident nodules only; the indeterminate nodules (hard_label == 1)
        are held out entirely into the test split with label ``-1`` (the
        ambiguous evaluation set). This is the recommended setup: the 3-class
        task with a trainable "uncertain" middle is not learnable.
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

    if binary:
        # benign(0)->0, malignant(2)->1, indeterminate(1)->-1 (held out)
        for r in records:
            r["binary_label"] = {0: 0, 2: 1}.get(int(r["hard_label"]), -1)
        tr, va, te = _binary_split(records, seed, ratios)
        label_key = "binary_label"
        class_names, num_classes = list(BINARY_CLASSES), 2
        amb_label = BINARY_AMBIGUOUS_LABEL
    else:
        tr, va, te = _stratified_split(records, seed, ratios)
        label_key = "hard_label"
        class_names, num_classes = list(LIDC_CLASSES), NUM_CLASSES
        amb_label = AMBIGUOUS_LABEL

    def subset(indices, train, normalize=True):
        idx = np.array(indices)
        return LIDCDataset(patches[idx], [records[i] for i in indices],
                           train=train, normalize=normalize, label_key=label_key)

    train_ds = subset(tr, True)
    val_ds = subset(va, False)
    test_ds = subset(te, False)
    # Push set: train images, UNnormalized [0,1] 3-channel, for prototype saving.
    push_ds = subset(tr, False, normalize=False)

    if binary:
        n_amb = sum(1 for i in te if records[i]["binary_label"] == -1)
        log(f"[lidc] binary mode: train={len(tr)} val={len(va)} "
            f"test={len(te)} (incl. {n_amb} held-out ambiguous nodules)")

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
        class_names=class_names, num_classes=num_classes,
        ambiguous_label=amb_label, binary=binary,
    )


def get_ambiguous_mask(labels: torch.Tensor) -> torch.Tensor:
    """Mark the formally-defined ambiguous class (label == 1) as True."""
    labels = torch.as_tensor(labels)
    return labels == AMBIGUOUS_LABEL


def _verify_pylidc() -> None:
    """Preflight: confirm pylidc + the DICOM path are wired up correctly.

    Checks the bundled annotation DB is queryable and that at least one scan's
    DICOM volume actually loads from the path in ~/.pylidcrc.
    """
    try:
        import pylidc as pl
    except Exception as e:
        print(f"[verify] FAIL: cannot import pylidc ({e}). "
              f"pip install pylidc pydicom"); return
    n_scans = pl.query(pl.Scan).count()
    print(f"[verify] pylidc annotation DB OK: {n_scans} scans "
          f"(expected 1018 for full LIDC-IDRI)")
    scan = pl.query(pl.Scan).first()
    if scan is None:
        print("[verify] FAIL: no scans in DB."); return
    print(f"[verify] first scan: patient_id={scan.patient_id}")
    try:
        vol = scan.to_volume()
        print(f"[verify] DICOM load OK: volume shape {vol.shape} "
              f"(your ~/.pylidcrc 'path' points at the images correctly)")
    except Exception as e:
        print(f"[verify] FAIL: scan.to_volume() errored ({e}). "
              f"Check the 'path' in ~/.pylidcrc points at the folder "
              f"containing the LIDC-IDRI-XXXX directories.")
        return
    n3 = sum(1 for c in scan.cluster_annotations() if len(c) >= 3)
    print(f"[verify] first scan has {n3} nodule(s) with >=3 annotators")
    print("[verify] PASS — ready to build the cache.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build/inspect the LIDC cache")
    p.add_argument("--data_dir", help="LIDC CACHE directory (output of extraction)")
    p.add_argument("--build", action="store_true", help="Run extraction (pylidc source)")
    p.add_argument("--from_luna22", metavar="LUNA_DIR", default=None,
                   help="Build the cache from LUNA22-ISMI patches in LUNA_DIR "
                        "(contains LIDC-IDRI_1176.npy + .nii.gz / .zip). No pylidc needed.")
    p.add_argument("--patch_size", type=int, default=224,
                   help="Output slice size for LUNA22 extraction (match backbone)")
    p.add_argument("--max_scans", type=int, default=None,
                   help="Limit scans processed for pylidc (use e.g. 20 for a quick test)")
    p.add_argument("--verify", action="store_true",
                   help="Only check pylidc + DICOM path are configured; no extraction")
    args = p.parse_args()

    if args.verify:
        _verify_pylidc()
        raise SystemExit(0)

    if not args.data_dir:
        p.error("--data_dir is required (the cache output directory)")
    if args.from_luna22:
        extract_luna22_dataset(args.from_luna22, args.data_dir,
                               patch_size=args.patch_size)
    elif args.build:
        extract_lidc_dataset(args.data_dir, max_scans=args.max_scans)
    bundle = get_lidc_dataloaders(args.data_dir, build_if_missing=args.build)
    for name, ds in (("train", bundle.train_dataset), ("val", bundle.val_dataset),
                     ("test", bundle.test_dataset)):
        labels = ds.get_all_labels()
        counts = torch.bincount(labels, minlength=NUM_CLASSES).tolist()
        print(f"  {name}: {len(ds)} nodules  dist={dict(zip(LIDC_CLASSES, counts))}")
