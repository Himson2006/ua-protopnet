"""
HAM10000 data pipeline for Uncertainty-Aware ProtoPNet (UA-ProtoPNet).

HAM10000 is a 7-class dermatoscopy dataset for skin-lesion classification.
This module loads the official image folders + metadata CSV, performs a
deterministic stratified train/val/test split, builds augmented PyTorch
DataLoaders, and exposes utilities for flagging the "ambiguous" class pairs
that the UA-ProtoPNet uncertainty machinery is evaluated on.

Expected on-disk layout (the standard Kaggle / ISIC HAM10000 download)::

    <data_dir>/
        HAM10000_metadata.csv
        HAM10000_images_part_1/   ISIC_xxxxxxx.jpg ...
        HAM10000_images_part_2/   ISIC_xxxxxxx.jpg ...

Design notes
------------
* **Split by lesion, not by image.** A single physical lesion (``lesion_id``)
  can appear as several images. Splitting at the image level leaks near-
  duplicate views of the same lesion across train/test and inflates accuracy
  and *especially* calibration metrics — fatal for an uncertainty paper. We
  therefore assign each *lesion* to exactly one split, stratified by class.
  Set ``split_by_lesion=False`` to reproduce a naive image-level split.
* **Deterministic.** All randomness (the split and the augmentation seeds) is
  driven by an explicit ``seed`` so train/val/test membership is reproducible
  across machines (Mac <-> Linux GPU server).
* **Fixed class order.** ``HAM_CLASSES`` is sorted alphabetically and never
  reordered, so class index ``i`` means the same thing everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The 7 HAM10000 diagnosis labels in a fixed, deterministic (alphabetical)
#: order. Index ``i`` <-> ``HAM_CLASSES[i]`` everywhere in the project.
HAM_CLASSES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(HAM_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(HAM_CLASSES)}
NUM_CLASSES: int = len(HAM_CLASSES)

#: Human-readable names for figures / prototype labels.
CLASS_FULL_NAMES: Dict[str, str] = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma (malignant)",
    "nv": "Melanocytic nevus (benign)",
    "vasc": "Vascular lesion",
}

#: ImageNet normalization (backbones are ImageNet-pretrained).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Ambiguous class pairs used as the "uncertain" evaluation proxy for HAM10000.
#: These are visually confusable in dermatology; the model is *expected* to be
#: more uncertain on samples from these classes.
DEFAULT_AMBIGUOUS_PAIRS: List[Tuple[str, str]] = [("mel", "nv"), ("mel", "bkl")]

#: A clearly-separable pair used as the control group (expect LOW uncertainty).
CONTROL_PAIR: Tuple[str, str] = ("df", "nv")

IMG_SIZE = 224
RESIZE_SIZE = 256  # resize then crop to IMG_SIZE


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #

def build_path_index(data_dir: str) -> Dict[str, str]:
    """Scan the HAM10000 image folders once and map ``image_id -> filepath``.

    The dataset ships images split across ``HAM10000_images_part_1`` and
    ``HAM10000_images_part_2`` (and some redistributions use a single folder),
    so we scan every subfolder and index by the file stem (the ``ISIC_xxxxxxx``
    id, matching the ``image_id`` column of the metadata CSV).

    Parameters
    ----------
    data_dir : str
        Root of the HAM10000 download.

    Returns
    -------
    dict
        ``{image_id: absolute_path}`` for every ``.jpg``/``.png`` found.
    """
    index: Dict[str, str] = {}
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            lower = fname.lower()
            if lower.endswith((".jpg", ".jpeg", ".png")):
                stem = os.path.splitext(fname)[0]
                # Prefer the first occurrence; HAM10000 ids are unique.
                index.setdefault(stem, os.path.join(root, fname))
    return index


def load_metadata(data_dir: str, metadata_csv: Optional[str] = None) -> pd.DataFrame:
    """Load the HAM10000 metadata CSV and attach integer labels + file paths.

    Parameters
    ----------
    data_dir : str
        Root of the HAM10000 download.
    metadata_csv : str, optional
        Explicit path to the metadata CSV. Defaults to
        ``<data_dir>/HAM10000_metadata.csv``.

    Returns
    -------
    pandas.DataFrame
        The metadata with two added columns: ``label`` (int class index) and
        ``path`` (absolute image path). Rows whose image file is missing on
        disk are dropped with a warning count.
    """
    if metadata_csv is None:
        metadata_csv = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.isfile(metadata_csv):
        raise FileNotFoundError(
            f"HAM10000 metadata not found at '{metadata_csv}'. "
            f"Pass --data_dir pointing at the folder containing "
            f"HAM10000_metadata.csv and the HAM10000_images_part_* folders."
        )

    df = pd.read_csv(metadata_csv)
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    # Map diagnosis -> integer label using the fixed class order.
    unknown = set(df["dx"].unique()) - set(CLASS_TO_IDX)
    if unknown:
        raise ValueError(f"Unexpected dx values in metadata: {unknown}")
    df = df.copy()
    df["label"] = df["dx"].map(CLASS_TO_IDX).astype(int)

    # Attach file paths; drop rows whose image is absent on disk.
    path_index = build_path_index(data_dir)
    df["path"] = df["image_id"].map(path_index)
    n_before = len(df)
    df = df[df["path"].notna()].reset_index(drop=True)
    n_missing = n_before - len(df)
    if n_missing:
        print(f"[ham10000] WARNING: {n_missing} metadata rows had no matching "
              f"image file and were dropped ({len(df)} remain).")
    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def stratified_lesion_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    split_by_lesion: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified 70/15/15 train/val/test split.

    Stratifies by class so each split preserves the (very imbalanced) class
    distribution. When ``split_by_lesion`` is True, the unit assigned to a
    split is the *lesion* (``lesion_id``), guaranteeing that no lesion's images
    appear in more than one split (prevents leakage). Each lesion is stratified
    by its diagnosis (a lesion has a single ``dx``).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_metadata` (must contain ``label``, ``lesion_id``).
    seed : int
        RNG seed controlling the deterministic shuffle.
    ratios : (float, float, float)
        Train/val/test fractions; must sum to ~1.0.
    split_by_lesion : bool
        If False, split at the image level instead (naive; allows leakage).

    Returns
    -------
    (train_df, val_df, test_df) : tuple of DataFrames
    """
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} -> {sum(ratios)}")
    rng = np.random.RandomState(seed)
    r_train, r_val, _ = ratios

    if split_by_lesion:
        # One row per lesion with its class; stratify lesions by class.
        lesion_df = df.groupby("lesion_id", sort=True)["label"].first().reset_index()
        unit_key, unit_table = "lesion_id", lesion_df
    else:
        unit_key, unit_table = "image_id", df[["image_id", "label"]].copy()

    train_units: List = []
    val_units: List = []
    test_units: List = []
    for cls in range(NUM_CLASSES):
        units = unit_table.loc[unit_table["label"] == cls, unit_key].to_numpy()
        units = units.copy()
        rng.shuffle(units)
        n = len(units)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        # Remainder goes to test so the three splits exactly partition the data.
        train_units.extend(units[:n_train])
        val_units.extend(units[n_train:n_train + n_val])
        test_units.extend(units[n_train + n_val:])

    train_set, val_set, test_set = set(train_units), set(val_units), set(test_units)
    train_df = df[df[unit_key].isin(train_set)].reset_index(drop=True)
    val_df = df[df[unit_key].isin(val_set)].reset_index(drop=True)
    test_df = df[df[unit_key].isin(test_set)].reset_index(drop=True)

    # Sanity: no leakage when splitting by lesion.
    if split_by_lesion:
        assert not (set(train_df["lesion_id"]) & set(test_df["lesion_id"])), \
            "lesion leakage between train and test!"
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_transforms(train: bool) -> transforms.Compose:
    """Build torchvision transforms.

    Train: resize -> random crop to 224 + horizontal/vertical flips + rotation
    (+/-15 deg) + color jitter, then ImageNet normalize.
    Val/Test: resize -> center-crop to 224, then ImageNet normalize.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


def get_push_transform() -> transforms.Compose:
    """Transform for the prototype-push loader: UNnormalized images in [0,1].

    ProtoPNet's push needs raw [0,1] images for saving visualizable prototype
    patches; normalization is applied separately via
    :func:`imagenet_preprocess` just before the forward pass.
    """
    return transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])


def imagenet_preprocess(batch: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize a batch of [0,1] images (used as the push preprocess fn)."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class HAM10000Dataset(Dataset):
    """PyTorch ``Dataset`` over a HAM10000 split.

    Each item is ``(image_tensor[3,224,224], label_int)``. The originating
    DataFrame row index is recoverable via :attr:`row_meta` for downstream
    prototype-metadata bookkeeping (Section 10).
    """

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        # Cache the columns we need as plain lists for fast __getitem__.
        self.paths: List[str] = self.df["path"].tolist()
        self.labels: List[int] = self.df["label"].tolist()
        # Lightweight per-sample metadata for prototype naming / analysis.
        meta_cols = [c for c in ("image_id", "dx", "dx_type", "localization",
                                 "age", "sex", "lesion_id") if c in self.df.columns]
        self.row_meta: List[dict] = self.df[meta_cols].to_dict("records")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]

    def get_all_labels(self) -> torch.Tensor:
        """Return all integer labels for this split as a 1-D LongTensor."""
        return torch.as_tensor(self.labels, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Ambiguous-pair masking
# --------------------------------------------------------------------------- #

def _pairs_to_class_indices(
    ambiguous_pairs: Sequence[Tuple[str, str]]
) -> List[int]:
    """Map a list of (name, name) class pairs to the set of class indices."""
    idxs: set = set()
    for a, b in ambiguous_pairs:
        for name in (a, b):
            if name not in CLASS_TO_IDX:
                raise ValueError(f"Unknown HAM10000 class name '{name}'. "
                                 f"Valid: {HAM_CLASSES}")
            idxs.add(CLASS_TO_IDX[name])
    return sorted(idxs)


def get_ambiguous_mask(
    labels: torch.Tensor,
    ambiguous_pairs: Sequence[Tuple[str, str]] = DEFAULT_AMBIGUOUS_PAIRS,
) -> torch.Tensor:
    """Boolean mask marking samples that belong to the ambiguous class pairs.

    For HAM10000 the "ambiguous class" is defined at *evaluation* time: a
    sample is flagged ambiguous if its label is one of the classes appearing in
    any pair of ``ambiguous_pairs`` (e.g. ``mel`` or ``nv`` for the canonical
    ``mel<->nv`` pair). These are the samples on which UA-ProtoPNet is expected
    to report higher uncertainty.

    Parameters
    ----------
    labels : torch.Tensor
        1-D integer label tensor (class indices into :data:`HAM_CLASSES`).
    ambiguous_pairs : sequence of (str, str)
        Class-name pairs. Defaults to ``[('mel','nv'), ('mel','bkl')]``.

    Returns
    -------
    torch.Tensor
        Boolean tensor, same shape as ``labels``, ``True`` for ambiguous samples.
    """
    labels = torch.as_tensor(labels)
    amb_idxs = torch.tensor(_pairs_to_class_indices(ambiguous_pairs),
                            dtype=labels.dtype, device=labels.device)
    return torch.isin(labels, amb_idxs)


# --------------------------------------------------------------------------- #
# Public bundle + factory
# --------------------------------------------------------------------------- #

@dataclass
class HAM10000Bundle:
    """Container returned by :func:`get_ham10000_dataloaders`."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    push_loader: DataLoader
    train_dataset: HAM10000Dataset
    val_dataset: HAM10000Dataset
    test_dataset: HAM10000Dataset
    push_dataset: HAM10000Dataset
    preprocess_input_function: object = staticmethod(imagenet_preprocess)
    class_names: List[str] = field(default_factory=lambda: list(HAM_CLASSES))
    num_classes: int = NUM_CLASSES
    ambiguous_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: list(DEFAULT_AMBIGUOUS_PAIRS))


def _maybe_subsample(df: pd.DataFrame, max_samples: Optional[int],
                     seed: int) -> pd.DataFrame:
    """Stratified subsample down to ~max_samples rows (for smoke tests).

    Iterates groups explicitly rather than ``groupby(...).apply(...)``: on
    pandas >= 2.2 the latter drops the grouping column ('label') from the
    result, which breaks the downstream split.
    """
    if max_samples is None or len(df) <= max_samples:
        return df
    frac = max_samples / len(df)
    # Keep at least one sample per class where possible.
    parts = []
    for _label, group in df.groupby("label", sort=True):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def get_ham10000_dataloaders(
    data_dir: str,
    train_batch_size: int = 80,
    eval_batch_size: int = 100,
    push_batch_size: int = 75,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    split_by_lesion: bool = True,
    max_samples: Optional[int] = None,
    metadata_csv: Optional[str] = None,
) -> HAM10000Bundle:
    """Build train/val/test DataLoaders for HAM10000.

    Parameters
    ----------
    data_dir : str
        Root of the HAM10000 download (contains the metadata CSV and the
        ``HAM10000_images_part_*`` folders).
    train_batch_size, eval_batch_size : int
        Batch sizes for the train loader vs. the val/test loaders.
    num_workers : int
        DataLoader workers. Use 0 on Mac for stability; 4 on the Linux server.
    pin_memory : bool
        Pin memory (set True only on CUDA).
    seed : int
        Controls the deterministic split (and subsampling).
    ratios : (float, float, float)
        Train/val/test fractions.
    split_by_lesion : bool
        Split by lesion to prevent leakage (default, recommended).
    max_samples : int, optional
        If set, stratified-subsample *each split's source pool* so the total is
        ~max_samples — used for fast local smoke tests (Section 11G).
    metadata_csv : str, optional
        Override path to the metadata CSV.

    Returns
    -------
    HAM10000Bundle
        Dataclass with the three loaders, their datasets, and class metadata.
    """
    df = load_metadata(data_dir, metadata_csv=metadata_csv)
    if max_samples is not None:
        df = _maybe_subsample(df, max_samples, seed)

    train_df, val_df, test_df = stratified_lesion_split(
        df, seed=seed, ratios=ratios, split_by_lesion=split_by_lesion)

    train_ds = HAM10000Dataset(train_df, get_transforms(train=True))
    val_ds = HAM10000Dataset(val_df, get_transforms(train=False))
    test_ds = HAM10000Dataset(test_df, get_transforms(train=False))
    # Push set: same images as train, but UNnormalized for prototype saving.
    push_ds = HAM10000Dataset(train_df, get_push_transform())

    # Deterministic shuffling of the train loader.
    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size,
                              shuffle=True, generator=g, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size,
                            shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size,
                             shuffle=False, **common)
    push_loader = DataLoader(push_ds, batch_size=push_batch_size,
                             shuffle=False, **common)

    return HAM10000Bundle(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        push_loader=push_loader,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        push_dataset=push_ds,
    )


# --------------------------------------------------------------------------- #
# Self-test / smoke check
# --------------------------------------------------------------------------- #

def _self_test(data_dir: str, max_samples: Optional[int] = None) -> None:
    """Quick end-to-end check: load metadata, split, pull one batch."""
    print(f"[ham10000] self-test on data_dir={data_dir!r}")
    bundle = get_ham10000_dataloaders(
        data_dir, train_batch_size=8, eval_batch_size=8,
        num_workers=0, max_samples=max_samples)

    for name, ds in (("train", bundle.train_dataset),
                     ("val", bundle.val_dataset),
                     ("test", bundle.test_dataset)):
        labels = ds.get_all_labels()
        counts = torch.bincount(labels, minlength=NUM_CLASSES).tolist()
        dist = {HAM_CLASSES[i]: counts[i] for i in range(NUM_CLASSES)}
        print(f"  {name:5s}: {len(ds):6d} images  dist={dist}")

    # No lesion leakage train<->test.
    tr_les = set(bundle.train_dataset.df["lesion_id"])
    te_les = set(bundle.test_dataset.df["lesion_id"])
    print(f"  lesion overlap train/test: {len(tr_les & te_les)} (expect 0)")

    # Pull one real batch through the transforms.
    imgs, lbls = next(iter(bundle.train_loader))
    print(f"  batch images: {tuple(imgs.shape)} dtype={imgs.dtype} "
          f"range=[{imgs.min():.2f},{imgs.max():.2f}]  labels={lbls.tolist()}")

    # Ambiguous-mask utility.
    mask = get_ambiguous_mask(lbls)
    print(f"  ambiguous mask (mel/nv/bkl) on this batch: {mask.tolist()}")
    print("[ham10000] self-test OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HAM10000 pipeline self-test")
    parser.add_argument("--data_dir", required=True,
                        help="Root folder of the HAM10000 download")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample for a quick smoke test")
    args = parser.parse_args()
    _self_test(args.data_dir, max_samples=args.max_samples)
