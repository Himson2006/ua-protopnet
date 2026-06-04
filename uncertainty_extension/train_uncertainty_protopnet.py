"""
Uncertainty-Aware ProtoPNet training (Section 4).

Wraps the original ProtoPNet architecture (``model.construct_PPNet``) and its
3-phase training schedule (warm-up -> joint -> push -> FC fine-tune), but makes
everything device-agnostic and adds the uncertainty terms to the joint-phase
loss::

    total = crs_ent * CE(or soft-CE)
          + clst    * cluster_cost
          + sep     * separation_cost
          + l1      * l1_cost
          + lambda_u   * uncertainty_calibration_loss
          + lambda_div * prototype_diversity_loss

The cluster/separation/L1 costs replicate ``train_and_test._train_or_test``
faithfully, but without the ``.cuda()`` / ``model.module`` assumptions so the
code runs on Mac CPU (smoke tests) and the Linux GPU server alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pp_compat import add_repo_to_path, forward_with_similarity, resolve_device, unwrap
from .uncertainty_loss import (
    prototype_diversity_loss,
    soft_label_cross_entropy,
    uncertainty_calibration_loss,
)
from .uncertainty_scores import compute_class_evidence

add_repo_to_path()
import model as ppnet_model  # noqa: E402  (the cloned repo's model.py)

try:
    import wandb  # noqa: F401
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


DEFAULT_COEFS = {"crs_ent": 1.0, "clst": 0.8, "sep": 0.08, "l1": 1e-4}


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #

def build_ua_protopnet(
    base_architecture: str = "resnet50",
    num_classes: int = 7,
    prototypes_per_class: int = 10,
    prototype_dim: int = 128,
    img_size: int = 224,
    prototype_activation_function: str = "log",
    add_on_layers_type: str = "regular",
    pretrained: bool = True,
) -> nn.Module:
    """Construct a ProtoPNet sized for ``num_classes`` x ``prototypes_per_class``.

    Returns the bare ``PPNet`` (not DataParallel-wrapped); wrap it with
    :func:`maybe_dataparallel` for multi-GPU.
    """
    num_prototypes = num_classes * prototypes_per_class
    prototype_shape = (num_prototypes, prototype_dim, 1, 1)
    return ppnet_model.construct_PPNet(
        base_architecture=base_architecture,
        pretrained=pretrained,
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type,
    )


def maybe_dataparallel(model: nn.Module, multi_gpu: bool) -> nn.Module:
    """Optionally wrap in DataParallel when multiple GPUs are requested."""
    if multi_gpu and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


# --------------------------------------------------------------------------- #
# Phase parameter freezing (device-agnostic versions of tnt.warm/joint/last)
# --------------------------------------------------------------------------- #

def _set_requires(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def set_phase(model: nn.Module, phase: str) -> None:
    """Freeze/unfreeze parameter groups for 'warm', 'joint', or 'last'."""
    base = unwrap(model)
    if phase == "warm":
        _set_requires(base.features, False)
        _set_requires(base.add_on_layers, True)
        base.prototype_vectors.requires_grad = True
        _set_requires(base.last_layer, True)
    elif phase == "joint":
        _set_requires(base.features, True)
        _set_requires(base.add_on_layers, True)
        base.prototype_vectors.requires_grad = True
        _set_requires(base.last_layer, True)
    elif phase == "last":
        _set_requires(base.features, False)
        _set_requires(base.add_on_layers, False)
        base.prototype_vectors.requires_grad = False
        _set_requires(base.last_layer, True)
    else:
        raise ValueError(f"unknown phase {phase!r}")


def build_optimizers(model: nn.Module, lrs: Optional[dict] = None) -> Dict[str, torch.optim.Optimizer]:
    """Build the warm / joint / last-layer optimizers used across phases."""
    base = unwrap(model)
    lrs = lrs or {}
    joint_lrs = lrs.get("joint", {"features": 1e-4, "add_on_layers": 3e-3,
                                  "prototype_vectors": 3e-3})
    warm_lrs = lrs.get("warm", {"add_on_layers": 3e-3, "prototype_vectors": 3e-3})
    last_lr = lrs.get("last", 1e-4)

    joint = torch.optim.Adam([
        {"params": base.features.parameters(), "lr": joint_lrs["features"], "weight_decay": 1e-3},
        {"params": base.add_on_layers.parameters(), "lr": joint_lrs["add_on_layers"], "weight_decay": 1e-3},
        {"params": base.prototype_vectors, "lr": joint_lrs["prototype_vectors"]},
    ])
    warm = torch.optim.Adam([
        {"params": base.add_on_layers.parameters(), "lr": warm_lrs["add_on_layers"], "weight_decay": 1e-3},
        {"params": base.prototype_vectors, "lr": warm_lrs["prototype_vectors"]},
    ])
    last = torch.optim.Adam([
        {"params": base.last_layer.parameters(), "lr": last_lr},
    ])
    return {"warm": warm, "joint": joint, "last": last}


# --------------------------------------------------------------------------- #
# ProtoPNet costs (faithful, device-agnostic)
# --------------------------------------------------------------------------- #

def compute_protopnet_costs(
    base: nn.Module,
    min_distances: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    use_l1_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster, separation, and L1 costs (class-specific), as in ProtoPNet.

    Note the sign convention matches ``settings.coefs`` where ``sep`` is applied
    as ``- sep * separation_cost`` in the original; here we return the raw
    ``separation_cost`` and the caller subtracts (i.e. uses a positive ``sep``
    coefficient with a minus sign), consistent with :data:`DEFAULT_COEFS`.
    """
    proto_shape = base.prototype_shape
    max_dist = proto_shape[1] * proto_shape[2] * proto_shape[3]
    pci = base.prototype_class_identity.to(device)          # [P, C]

    correct = pci[:, labels].t()                             # [B, P]
    inverted, _ = torch.max((max_dist - min_distances) * correct, dim=1)
    cluster_cost = torch.mean(max_dist - inverted)

    wrong = 1 - correct
    inverted_w, _ = torch.max((max_dist - min_distances) * wrong, dim=1)
    separation_cost = torch.mean(max_dist - inverted_w)

    if use_l1_mask:
        l1_mask = 1 - pci.t()                                # [C, P]
        l1 = (base.last_layer.weight * l1_mask).norm(p=1)
    else:
        l1 = base.last_layer.weight.norm(p=1)
    return cluster_cost, separation_cost, l1


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #

@dataclass
class EpochStats:
    loss: float = 0.0
    accuracy: float = 0.0
    cross_entropy: float = 0.0
    cluster: float = 0.0
    separation: float = 0.0
    l1: float = 0.0
    uncertainty: float = 0.0
    diversity: float = 0.0


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    coefs: dict,
    lambda_u: float = 0.5,
    lambda_div: float = 0.01,
    temperature: float = 1.0,
    ambiguous_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_soft_labels: bool = False,
    enforce_certainty_on_clear: bool = True,
    add_uncertainty: bool = True,
) -> EpochStats:
    """Run one train (optimizer given) or eval (optimizer None) epoch.

    ``add_uncertainty`` is False in warm-up / last-layer phases (per Section 4,
    the uncertainty terms are added in the joint phase). ``use_soft_labels``
    swaps CE for soft-label CE on LIDC (loader yields 4-tuples).
    """
    is_train = optimizer is not None
    model.train(is_train)
    base = unwrap(model)
    num_classes = base.num_classes
    K = base.num_prototypes // num_classes

    stats = EpochStats()
    n_examples = 0
    n_correct = 0
    n_batches = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            images = batch[0].to(device)
            hard_labels = batch[1].to(device)
            soft_labels = (batch[2].to(device)
                           if use_soft_labels and len(batch) >= 3 else None)

            logits, min_distances, sims = forward_with_similarity(model, images)

            if use_soft_labels and soft_labels is not None:
                ce = soft_label_cross_entropy(logits, soft_labels)
            else:
                ce = F.cross_entropy(logits, hard_labels)

            cluster, separation, l1 = compute_protopnet_costs(
                base, min_distances, hard_labels, device)

            loss = (coefs["crs_ent"] * ce
                    + coefs["clst"] * cluster
                    - coefs["sep"] * separation
                    + coefs["l1"] * l1)

            u_loss = torch.zeros((), device=device)
            div_loss = torch.zeros((), device=device)
            if add_uncertainty:
                class_evidence = compute_class_evidence(sims, num_classes, K)
                amb_mask = (ambiguous_fn(hard_labels) if ambiguous_fn is not None
                            else torch.zeros_like(hard_labels, dtype=torch.bool))
                u_loss = uncertainty_calibration_loss(
                    class_evidence, hard_labels, amb_mask, temperature=temperature,
                    enforce_certainty_on_clear=enforce_certainty_on_clear)
                div_loss = prototype_diversity_loss(
                    base.prototype_vectors, base.prototype_class_identity)
                loss = loss + lambda_u * u_loss + lambda_div * div_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            n_correct += (preds == hard_labels).sum().item()
            n_examples += hard_labels.size(0)
            n_batches += 1
            stats.loss += loss.item()
            stats.cross_entropy += ce.item()
            stats.cluster += cluster.item()
            stats.separation += separation.item()
            stats.l1 += l1.item()
            stats.uncertainty += float(u_loss)
            stats.diversity += float(div_loss)

    if n_batches:
        for fld in ("loss", "cross_entropy", "cluster", "separation", "l1",
                    "uncertainty", "diversity"):
            setattr(stats, fld, getattr(stats, fld) / n_batches)
    stats.accuracy = n_correct / max(1, n_examples)
    return stats


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(model, optimizers, epoch, phase, ckpt_dir, extra=None):
    """Save a resumable checkpoint."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "phase": phase,
        "model_state": unwrap(model).state_dict(),
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "extra": extra or {},
    }
    path = os.path.join(ckpt_dir, "latest.pth")
    torch.save(state, path)
    return path


def load_checkpoint(model, optimizers, ckpt_dir, device):
    """Load the latest checkpoint if present; return the stored dict or None."""
    path = os.path.join(ckpt_dir, "latest.pth")
    if not os.path.isfile(path):
        return None
    state = torch.load(path, map_location=device)
    unwrap(model).load_state_dict(state["model_state"])
    if optimizers and "optimizers" in state:
        for k, o in optimizers.items():
            if k in state["optimizers"]:
                o.load_state_dict(state["optimizers"][k])
    return state


# --------------------------------------------------------------------------- #
# Full training driver
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs_warm: int = 5
    epochs_joint: int = 25
    epochs_last: int = 10
    coefs: dict = field(default_factory=lambda: dict(DEFAULT_COEFS))
    lambda_u: float = 0.5
    lambda_div: float = 0.01
    temperature: float = 1.0
    use_soft_labels: bool = False
    enforce_certainty_on_clear: bool = True
    joint_lr_step_size: int = 5
    joint_lr_gamma: float = 0.1
    ckpt_dir: Optional[str] = None
    push_dir: Optional[str] = None
    log: Callable = print


def train_ua_protopnet(
    model: nn.Module,
    train_loader,
    val_loader,
    push_loader,
    device: torch.device,
    config: TrainConfig,
    ambiguous_fn: Optional[Callable] = None,
    preprocess_input_function: Optional[Callable] = None,
    eval_fn: Optional[Callable] = None,
    resume: bool = False,
    wandb_run=None,
) -> Dict[str, List]:
    """Run the full warm -> joint(+push) -> FC-finetune schedule.

    Parameters
    ----------
    push_loader : DataLoader
        Unnormalized [0,1] loader used for the prototype push.
    eval_fn : callable, optional
        ``eval_fn(model) -> dict`` for richer per-epoch metrics (AUROC, ECE,
        ambiguous/clear). Typically a closure over
        :func:`evaluate.run_full_evaluation`.
    Returns
    -------
    dict
        History of per-epoch stats/metrics.
    """
    log = config.log
    optimizers = build_optimizers(model)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizers["joint"], step_size=config.joint_lr_step_size,
        gamma=config.joint_lr_gamma)
    history: Dict[str, List] = {"train": [], "val": [], "metrics": []}

    start_epoch = 0
    if resume and config.ckpt_dir:
        state = load_checkpoint(model, optimizers, config.ckpt_dir, device)
        if state is not None:
            start_epoch = state["epoch"] + 1
            log(f"[train] resumed from epoch {start_epoch}")

    def _log_epoch(tag, epoch, stats: EpochStats, metrics=None):
        log(f"[{tag}] epoch {epoch}: loss={stats.loss:.4f} "
            f"acc={stats.accuracy:.4f} ce={stats.cross_entropy:.4f} "
            f"clst={stats.cluster:.3f} sep={stats.separation:.3f} "
            f"u={stats.uncertainty:.4f} div={stats.diversity:.4f}")
        if metrics:
            avc = metrics.get("ambiguous_vs_clear", {})
            log(f"        val_acc={metrics.get('accuracy', float('nan')):.4f} "
                f"ECE={metrics.get('ece', float('nan')):.4f} "
                f"AUROC={metrics.get('uncertainty_auroc', float('nan')):.3f} "
                f"U_amb={avc.get('u_ambiguous', float('nan')):.3f} "
                f"U_clear={avc.get('u_clear', float('nan')):.3f} "
                f"ratio={avc.get('ratio', float('nan')):.2f}")
        if wandb_run is not None:
            payload = {f"{tag}/{k}": getattr(stats, k) for k in
                       ("loss", "accuracy", "cross_entropy", "cluster",
                        "separation", "uncertainty", "diversity")}
            if metrics:
                payload.update({f"val/{k}": v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            wandb_run.log(payload, step=epoch)

    global_epoch = start_epoch
    total = config.epochs_warm + config.epochs_joint

    # ---- Phase 1 + 2: warm-up then joint -------------------------------- #
    for epoch in range(start_epoch, total):
        is_warm = epoch < config.epochs_warm
        phase = "warm" if is_warm else "joint"
        set_phase(model, phase)
        opt = optimizers[phase]
        tr = run_epoch(
            model, train_loader, device, opt, config.coefs,
            lambda_u=config.lambda_u, lambda_div=config.lambda_div,
            temperature=config.temperature, ambiguous_fn=ambiguous_fn,
            use_soft_labels=config.use_soft_labels,
            enforce_certainty_on_clear=config.enforce_certainty_on_clear,
            add_uncertainty=not is_warm)
        if not is_warm:
            scheduler.step()
        metrics = eval_fn(model) if eval_fn is not None else None
        history["train"].append(tr.__dict__)
        history["metrics"].append(metrics)
        _log_epoch(phase, global_epoch, tr, metrics)
        if config.ckpt_dir:
            save_checkpoint(model, optimizers, global_epoch, phase, config.ckpt_dir)
        global_epoch += 1

    # ---- Phase 3: prototype push ---------------------------------------- #
    from .push_uncertainty import push_prototypes
    proto_source = push_prototypes(
        push_loader, model, device,
        preprocess_input_function=preprocess_input_function,
        root_dir_for_saving_prototypes=config.push_dir,
        epoch_number=global_epoch,
        save_images=config.push_dir is not None,
        log=log)
    history["prototype_source"] = proto_source

    # ---- Phase 4: FC fine-tune ------------------------------------------ #
    set_phase(model, "last")
    for i in range(config.epochs_last):
        tr = run_epoch(
            model, train_loader, device, optimizers["last"], config.coefs,
            lambda_u=config.lambda_u, lambda_div=config.lambda_div,
            temperature=config.temperature, ambiguous_fn=ambiguous_fn,
            use_soft_labels=config.use_soft_labels,
            add_uncertainty=False)  # only the FC layer trains here
        metrics = eval_fn(model) if eval_fn is not None else None
        history["train"].append(tr.__dict__)
        history["metrics"].append(metrics)
        _log_epoch("last", global_epoch, tr, metrics)
        if config.ckpt_dir:
            save_checkpoint(model, optimizers, global_epoch, "last", config.ckpt_dir)
        global_epoch += 1

    return history
