# src/datasets/ucf101_dataloader.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.transforms.functional import crop

from utils.dataset_wrappers import (
    VideoLabelDataset,
    FractionalClassSubset,
)  # your thin wrapper
from functools import partial

# constants
CROP_SIZE = 224
_MEAN: List[float] = [0.45, 0.45, 0.45]  # Kinetics‑400 RGB mean
_STD: List[float] = [0.225, 0.225, 0.225]  # Kinetics‑400 RGB std


# helper functions


def fix_num_frames_transform(video: torch.Tensor, n: int) -> torch.Tensor:
    """
    Pad or truncate video to exactly n frames (expects uint8 [T,H,W,C]).
    """
    t = video.size(0)
    if t >= n:
        return video[:n]
    pad = video[-1:].repeat(n - t, 1, 1, 1)
    return torch.cat([video, pad], 0)  # [n,H,W,C]


def to_float_tchw_transform(video: torch.Tensor) -> torch.Tensor:
    """
    uint8 [T,H,W,C] -> float32 [T,C,H,W] in [0,1].
    """
    return video.float().div_(255.0).permute(0, 3, 1, 2)


def crop_and_flip_transform(clip: torch.Tensor, train: bool) -> torch.Tensor:
    """
    Apply one crop (+/- flip) identically to all frames.
    Input/Output: [T,C,H,W].
    """
    t, c, h, w = clip.shape
    if train:
        scale = 0.8 + 0.2 * torch.rand(1).item()  # random scale [0.8..1.0]
        oh = ow = int(min(h, w) * scale)
        i = torch.randint(0, h - oh + 1, ()).item()
        j = torch.randint(0, w - ow + 1, ()).item()
        do_flip = torch.rand(1).item() < 0.5
    else:
        oh = ow = min(h, w)
        i = (h - oh) // 2
        j = (w - ow) // 2
        do_flip = False

    frames = []
    for k in range(t):
        f = crop(clip[k], i, j, oh, ow)  # [C, oh, ow]
        if do_flip:
            f = f.flip(-1)
        frames.append(f)
    clip = torch.stack(frames, 0)  # [T,C,oh,ow]

    # resize to CROP_SIZE
    clip = F.interpolate(clip, size=CROP_SIZE, mode="bilinear", align_corners=False)
    return clip


def permute_cthw_transform(video: torch.Tensor) -> torch.Tensor:
    """
    Permute from [T,C,H,W] -> [C,T,H,W].
    """
    return video.permute(1, 0, 2, 3)


# main transform builder


def _video_transform(train: bool, n_frames: int) -> Compose:
    """
    Builds a Compose of top-level callables,
    returning float32 [C,T,H,W] ready for 3-D CNN.
    """
    return Compose(
        [
            # fix # frames
            Lambda(partial(fix_num_frames_transform, n=n_frames)),
            # convert to float TCHW
            Lambda(to_float_tchw_transform),
            # random or center crop + optional flip
            Lambda(partial(crop_and_flip_transform, train=train)),
            # normalize
            Normalize(_MEAN, _STD),
            # permute to [C,T,H,W]
            Lambda(permute_cthw_transform),
        ]
    )


# dataset / dataloader factories


def get_ucf101_datasets(
    root: str | Path,
    annotation_path: str | Path,
    *,
    fold: int = 1,
    frames_per_clip: int = 16,
    wrap: bool = True,
    fraction: float = 0.25,
) -> Tuple[UCF101, UCF101]:
    root, ann = Path(root), Path(annotation_path)

    train_raw = UCF101(
        root=root,
        annotation_path=ann,
        fold=fold,
        train=True,
        frames_per_clip=frames_per_clip,
        step_between_clips=frames_per_clip,
        transform=_video_transform(True, frames_per_clip),
    )

    val_raw = UCF101(
        root=root,
        annotation_path=ann,
        fold=fold,
        train=False,
        frames_per_clip=frames_per_clip,
        step_between_clips=frames_per_clip,
        transform=_video_transform(False, frames_per_clip),
    )

    train_fraction_ds = FractionalClassSubset(
        train_raw, fraction=fraction, min_videos=1
    )
    val_fraction_ds = FractionalClassSubset(val_raw, fraction=fraction, min_videos=1)
    if wrap:
        return VideoLabelDataset(train_raw), VideoLabelDataset(val_raw)
        # return VideoLabelDataset(train_fraction_ds), VideoLabelDataset(val_fraction_ds)
    return train_raw, val_raw


def get_ucf101_loaders(
    root: str | Path,
    annotation_path: str | Path,
    *,
    fold: int = 1,
    frames_per_clip: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = get_ucf101_datasets(
        root, annotation_path, fold=fold, frames_per_clip=frames_per_clip
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # can be > 0 now
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ld, val_ld
