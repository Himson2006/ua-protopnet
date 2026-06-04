"""
Compatibility / glue layer between UA-ProtoPNet and the original ProtoPNet repo.

The original repo (``model.py``, ``push.py``, ``train_and_test.py``) assumes a
single CUDA device and ``torch.nn.DataParallel``-wrapped models (it accesses
``model.module`` and hardcodes ``.cuda()``). The uncertainty extension must run
on Mac CPU (smoke tests) *and* a Linux GPU server. This module centralizes:

* putting the cloned repo root on ``sys.path`` so ``import model`` works,
* resolving the compute device from a CLI string,
* unwrapping a (possibly ``DataParallel``) model to the underlying ``PPNet``,
* extracting prototype *similarity* scores from a forward pass.

Keeping these in one place means the rest of the extension never calls
``.cuda()`` or touches ``.module`` directly.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Repo path bootstrap
# --------------------------------------------------------------------------- #

#: Absolute path to the cloned ProtoPNet repo root (the parent of this package).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_repo_to_path() -> str:
    """Ensure the ProtoPNet repo root is importable; return it.

    Lets us ``import model``, ``import push``, ``import train_and_test`` etc.
    from the original repo regardless of the current working directory.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #

def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device string to a ``torch.device``, auto-detecting if None.

    Preference order when ``device`` is None or ``'auto'``: CUDA, then Apple
    MPS, then CPU.
    """
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Model unwrapping / forward
# --------------------------------------------------------------------------- #

def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying ``PPNet`` whether or not it is ``DataParallel``-wrapped."""
    return model.module if isinstance(model, nn.DataParallel) else model


def forward_with_similarity(
    model: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a forward pass and also return prototype similarity activations.

    The original ``PPNet.forward`` returns ``(logits, min_distances)``; the
    prototype *similarity* (activation) is ``distance_2_similarity(min_distances)``
    — exactly the per-prototype activation the FC layer consumes, and the input
    our uncertainty functions expect.

    Returns
    -------
    (logits, min_distances, similarities) : tuple of torch.Tensor
        ``logits``        : ``[batch, num_classes]``
        ``min_distances`` : ``[batch, num_prototypes]``
        ``similarities``  : ``[batch, num_prototypes]`` prototype activations
    """
    logits, min_distances = model(x)
    similarities = unwrap(model).distance_2_similarity(min_distances)
    return logits, min_distances, similarities


def get_prototypes_per_class(model: nn.Module) -> int:
    """Return K = num_prototypes // num_classes for the given PPNet."""
    base = unwrap(model)
    return base.num_prototypes // base.num_classes
