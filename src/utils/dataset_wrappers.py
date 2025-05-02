import torch
from torch.utils.data import Dataset
import os
import math
import random


class VideoLabelDataset(Dataset):
    """
    Wraps a torchvision UCF101 dataset so that each sample is
    returned as (video, label), i.e. drops the audio track.
    """

    def __init__(self, base_ds: Dataset):
        self.base = base_ds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, _audio, label = self.base[idx]  # unpack 4‑tuple
        return video, label  # keep the 2 we need


# ── tiny wrapper so each sample is (X, y) for Skorch ──
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]  # dataset returns (video, label)
        return x, y


import os
import math
from torch.utils.data import Dataset


class FractionalClassSubset(Dataset):
    """
    Wrapper for the TorchVision UCF101 dataset to keep the same
    distribution of classes, but only a chosen fraction (0..1) of each.
    """

    def __init__(self, base_ds, fraction=0.25, min_videos=1):
        super().__init__()
        self.base = base_ds
        self.fraction = fraction
        self.min_videos = min_videos
        self.indices = []

        # Find out how many videos each class has
        class_to_indices = {}
        for i in range(len(base_ds.samples)):
            video_path = base_ds.samples[i][0]
            cls_name = os.path.basename(os.path.dirname(video_path))

            if cls_name not in class_to_indices:
                class_to_indices[cls_name] = []
            class_to_indices[cls_name].append(i)

        # For each class, keep only fraction * #videos (rounded)
        for cls_name, idx_list in class_to_indices.items():
            total = len(idx_list)
            keep_count = max(self.min_videos, math.floor(total * self.fraction))
            if keep_count > total:
                keep_count = total  # if fraction is > 1 or if class is small

            # Picking random 'keep_count' videos from this class
            # random.sample does the shuffle + slice in one g0
            selected = random.sample(idx_list, keep_count)
            self.indices.extend(selected)

        self.indices = sorted(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
