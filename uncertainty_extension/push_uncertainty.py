"""
Device-agnostic prototype push for UA-ProtoPNet.

The original ``push.push_prototypes`` hardcodes ``.cuda()`` and ``model.module``
(DataParallel). This reimplementation does the same job — replace each
prototype with the nearest real training-patch representation and save the
visualizable prototype images — but runs on CPU / MPS / CUDA and on a bare
``PPNet``. It deliberately preserves the original on-disk layout so the
visualization module (Section 6) can load prototypes unchanged::

    <root>/epoch-<N>/
        prototype-img<j>.png                   # high-activation crop (THE patch)
        prototype-img-original<j>.png          # full source image
        prototype-img-original_with_self_act<j>.png
        prototype-self-act<j>.npy              # activation map
        bb<N>.npy                              # [j] -> [img_idx, y0,y1,x0,x1, cls]

It also returns a per-prototype source record (image index + class) that
:mod:`run_experiment` joins with dataset metadata to build
``prototype_metadata.json`` (Section 10).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .pp_compat import add_repo_to_path, unwrap

add_repo_to_path()
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from helpers import find_high_activation_crop, makedir  # noqa: E402  (repo)
from receptive_field import compute_rf_prototype  # noqa: E402  (repo)


@torch.no_grad()
def push_prototypes(
    push_loader,
    model,
    device,
    class_specific: bool = True,
    preprocess_input_function: Optional[Callable] = None,
    prototype_layer_stride: int = 1,
    root_dir_for_saving_prototypes: Optional[str] = None,
    epoch_number: Optional[int] = None,
    prototype_img_filename_prefix: str = "prototype-img",
    prototype_self_act_filename_prefix: str = "prototype-self-act",
    proto_bound_boxes_filename_prefix: str = "bb",
    save_images: bool = True,
    log: Callable = print,
) -> List[Dict]:
    """Push every prototype to its nearest training patch; optionally save images.

    Parameters
    ----------
    push_loader : DataLoader
        Yields **unnormalized** images in ``[0, 1]`` (first element of each
        batch) and integer labels (second element). 4-tuples (LIDC) are fine;
        only the first two elements are used.
    model : nn.Module
        The PPNet (or DataParallel-wrapped PPNet).
    device : torch.device
        Compute device.
    preprocess_input_function : callable, optional
        Applied to the unnormalized batch before the forward pass (e.g. ImageNet
        normalization). The *saved* images use the unnormalized input.
    root_dir_for_saving_prototypes : str, optional
        If given (and ``save_images``), images/arrays are written under
        ``<root>/epoch-<epoch_number>/``.
    save_images : bool
        Set False for fast smoke tests (still updates prototype vectors).

    Returns
    -------
    list of dict
        One record per prototype: ``{proto_idx, class_id, source_image_index,
        min_distance}``. ``source_image_index`` indexes into the push dataset.
    """
    base = unwrap(model)
    base.eval()
    log("\tpush (device-agnostic)")

    proto_shape = base.prototype_shape
    n_protos = base.num_prototypes
    num_classes = base.num_classes
    proto_h, proto_w = proto_shape[2], proto_shape[3]

    global_min_dist = np.full(n_protos, np.inf)
    global_min_patches = np.zeros([n_protos, proto_shape[1], proto_h, proto_w],
                                  dtype=np.float32)
    # Bookkeeping for saving + metadata.
    proto_bound_boxes = np.full([n_protos, 6], -1, dtype=np.int64)
    proto_source = [
        {"proto_idx": j,
         "class_id": int(torch.argmax(base.prototype_class_identity[j]).item()),
         "source_image_index": -1, "min_distance": float("inf")}
        for j in range(n_protos)
    ]
    # Cache the winning source image + activation map for later saving.
    saved_blob: Dict[int, dict] = {}

    search_batch_size = push_loader.batch_size or 1

    out_dir = None
    if root_dir_for_saving_prototypes is not None and save_images:
        out_dir = (os.path.join(root_dir_for_saving_prototypes,
                                f"epoch-{epoch_number}")
                   if epoch_number is not None
                   else root_dir_for_saving_prototypes)
        makedir(out_dir)

    for push_iter, batch in enumerate(push_loader):
        search_unnorm = batch[0]                       # [B,3,H,W] in [0,1]
        search_y = batch[1]
        start_idx = push_iter * search_batch_size

        batch_in = (preprocess_input_function(search_unnorm)
                    if preprocess_input_function is not None else search_unnorm)
        conv_output, distances = base.push_forward(batch_in.to(device))
        conv_output = conv_output.detach().cpu().numpy()
        distances = distances.detach().cpu().numpy()   # [B, P, h, w]

        # Map class -> image indices in this batch (for class-specific search).
        cls_to_imgs: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
        for img_i, y in enumerate(search_y):
            cls_to_imgs[int(y.item())].append(img_i)

        for j in range(n_protos):
            if class_specific:
                target_cls = proto_source[j]["class_id"]
                img_idxs = cls_to_imgs[target_cls]
                if not img_idxs:
                    continue
                dist_j = distances[img_idxs][:, j, :, :]
            else:
                img_idxs = list(range(distances.shape[0]))
                dist_j = distances[:, j, :, :]

            batch_min = float(np.amin(dist_j))
            if batch_min >= global_min_dist[j]:
                continue

            argmin = list(np.unravel_index(np.argmin(dist_j, axis=None),
                                           dist_j.shape))
            img_in_batch = img_idxs[argmin[0]]
            hs = argmin[1] * prototype_layer_stride
            ws = argmin[2] * prototype_layer_stride
            patch = conv_output[img_in_batch, :, hs:hs + proto_h, ws:ws + proto_w]

            global_min_dist[j] = batch_min
            global_min_patches[j] = patch
            proto_source[j]["source_image_index"] = int(start_idx + img_in_batch)
            proto_source[j]["min_distance"] = batch_min

            # Stash everything needed to render this prototype's images later.
            saved_blob[j] = {
                "orig_img": np.transpose(
                    search_unnorm[img_in_batch].cpu().numpy(), (1, 2, 0)),
                "dist_map": distances[img_in_batch, j, :, :],
                "argmin": argmin,
                "img_size": search_unnorm.shape[2],
                "global_img_index": int(start_idx + img_in_batch),
                "class_id": proto_source[j]["class_id"],
            }

    # --- write the new prototype vectors back into the model ---------------- #
    update = np.reshape(global_min_patches, tuple(proto_shape))
    base.prototype_vectors.data.copy_(
        torch.tensor(update, dtype=torch.float32, device=base.prototype_vectors.device))

    # --- save visualizable prototype images --------------------------------- #
    if out_dir is not None:
        for j, blob in saved_blob.items():
            _save_prototype_images(
                base, j, blob, out_dir,
                prototype_img_filename_prefix,
                prototype_self_act_filename_prefix,
                proto_bound_boxes)
        np.save(os.path.join(out_dir,
                             f"{proto_bound_boxes_filename_prefix}{epoch_number}.npy"),
                proto_bound_boxes)

    log(f"\tpush done: {sum(np.isfinite(global_min_dist))}/{n_protos} prototypes updated")
    return proto_source


def _save_prototype_images(base, j, blob, out_dir, img_prefix, act_prefix,
                           proto_bound_boxes) -> None:
    """Render and save the images/arrays for a single prototype j."""
    orig = blob["orig_img"]
    img_size = blob["img_size"]
    dist_map = blob["dist_map"]

    # distance -> activation (matches PPNet.distance_2_similarity in numpy)
    if base.prototype_activation_function == "log":
        act = np.log((dist_map + 1) / (dist_map + base.epsilon))
    elif base.prototype_activation_function == "linear":
        max_dist = base.prototype_shape[1] * base.prototype_shape[2] * base.prototype_shape[3]
        act = max_dist - dist_map
    else:
        act = dist_map  # fallback; custom activations handled upstream
    up = cv2.resize(act, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # High-activation crop = the prototype patch.
    bound = find_high_activation_crop(up)
    proto_img = orig[bound[0]:bound[1], bound[2]:bound[3], :]

    proto_bound_boxes[j, 0] = blob["global_img_index"]
    proto_bound_boxes[j, 1:5] = bound
    proto_bound_boxes[j, 5] = blob["class_id"]

    np.save(os.path.join(out_dir, f"{act_prefix}{j}.npy"), act)
    plt.imsave(os.path.join(out_dir, f"{img_prefix}-original{j}.png"),
               np.clip(orig, 0, 1), vmin=0.0, vmax=1.0)
    plt.imsave(os.path.join(out_dir, f"{img_prefix}{j}.png"),
               np.clip(proto_img, 0, 1), vmin=0.0, vmax=1.0)

    # Overlay of activation heatmap on the source image.
    rescaled = up - np.amin(up)
    denom = np.amax(rescaled)
    if denom > 0:
        rescaled = rescaled / denom
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap[..., ::-1]) / 255.0
    overlay = 0.5 * np.clip(orig, 0, 1) + 0.3 * heatmap
    plt.imsave(os.path.join(out_dir, f"{img_prefix}-original_with_self_act{j}.png"),
               np.clip(overlay, 0, 1), vmin=0.0, vmax=1.0)
