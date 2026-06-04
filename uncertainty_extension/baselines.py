"""
Uncertainty baselines for comparison (Section 7).

Two standard scalar-uncertainty methods wrapped around ProtoPNet so they share
the exact evaluation interface used by the main model
(:func:`evaluate.run_full_evaluation` via its ``forward_fn`` hook):

* :class:`MCDropoutProtoPNet` — Monte-Carlo dropout at the prototype-activation
  layer; uncertainty = variance of class probabilities across stochastic passes.
* :class:`EnsembleProtoPNet` — average of several independently-trained
  ProtoPNets; uncertainty = variance of class probabilities across members.

Crucially both produce a *scalar* uncertainty only. Neither can name *which
prototypes compete* to cause the uncertainty — that interpretable explanation
is what UA-ProtoPNet adds. Each exposes ``forward_fn`` returning
``(logits, class_evidence, uncertainty)`` so the same evaluation pass runs on
all methods.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pp_compat import unwrap
from .uncertainty_scores import compute_class_evidence


class MCDropoutProtoPNet(nn.Module):
    """Monte-Carlo Dropout inference on top of a trained ProtoPNet.

    Dropout is applied to the prototype *activations* (after the prototype
    layer, before the FC layer) and left active at inference; ``n_samples``
    stochastic passes give a distribution over class probabilities. The scalar
    uncertainty is the mean (over classes) per-sample variance of those
    probabilities.

    Parameters
    ----------
    ppnet : nn.Module
        A trained PPNet (or DataParallel-wrapped).
    n_samples : int
        Number of stochastic forward passes (default 30).
    dropout_p : float
        Dropout probability.
    """

    def __init__(self, ppnet: nn.Module, n_samples: int = 30, dropout_p: float = 0.2):
        super().__init__()
        self.ppnet = ppnet
        self.n_samples = n_samples
        self.dropout_p = dropout_p

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        """Return ``(mean_logits, class_evidence, uncertainty)`` for a batch."""
        base = unwrap(self.ppnet)
        num_classes = base.num_classes
        K = base.num_prototypes // num_classes

        _logits, min_distances = self.ppnet(images)
        sims = base.distance_2_similarity(min_distances)   # [B, P] activations

        prob_samples = []
        for _ in range(self.n_samples):
            dropped = F.dropout(sims, p=self.dropout_p, training=True)
            logits = base.last_layer(dropped)
            prob_samples.append(F.softmax(logits, dim=1))
        stack = torch.stack(prob_samples, dim=0)           # [S, B, C]
        mean_probs = stack.mean(dim=0)                     # [B, C]
        # Scalar uncertainty: mean over classes of across-sample variance.
        uncertainty = stack.var(dim=0).mean(dim=1)         # [B]
        mean_logits = torch.log(mean_probs + 1e-12)
        # class_evidence still from the deterministic similarity pass (so the
        # shared metrics that need per-class evidence remain defined).
        class_evidence = compute_class_evidence(sims, num_classes, K)
        return mean_logits, class_evidence, uncertainty

    def forward_fn(self) -> Callable:
        """Return an ``(model, images) -> (...)`` closure for run_full_evaluation."""
        return lambda _model, images: self.predict(images)


class EnsembleProtoPNet(nn.Module):
    """Deep-ensemble of independently-trained ProtoPNets.

    Averages class probabilities across members; the scalar uncertainty is the
    mean (over classes) per-sample variance of member probabilities. Class
    evidence is averaged across members for the interpretable-but-shared metrics.

    Parameters
    ----------
    models : list of nn.Module
        Trained PPNets (typically 5, different seeds). All must share
        ``num_classes`` and prototype allocation.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("EnsembleProtoPNet requires >= 1 model")
        self.models = nn.ModuleList(models)
        base0 = unwrap(models[0])
        self.num_classes = base0.num_classes
        self.K = base0.num_prototypes // self.num_classes

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        prob_members = []
        evidence_members = []
        for m in self.models:
            base = unwrap(m)
            logits, min_distances = m(images)
            sims = base.distance_2_similarity(min_distances)
            prob_members.append(F.softmax(logits, dim=1))
            evidence_members.append(
                compute_class_evidence(sims, self.num_classes, self.K))
        probs = torch.stack(prob_members, dim=0)           # [M, B, C]
        mean_probs = probs.mean(dim=0)
        uncertainty = probs.var(dim=0).mean(dim=1)         # [B]
        mean_logits = torch.log(mean_probs + 1e-12)
        class_evidence = torch.stack(evidence_members, dim=0).mean(dim=0)
        return mean_logits, class_evidence, uncertainty

    def forward_fn(self) -> Callable:
        return lambda _model, images: self.predict(images)
