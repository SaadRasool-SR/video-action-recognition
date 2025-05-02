"""
main.py (Generated from main_base.py)

Description:
    A minimal CLI entry point for your project.
    Demonstrates basic usage of Python's argparse module.
"""

import argparse
import sys
import torch
import torchvision
import torchaudio
import numpy as np
import pandas
import scipy
import matplotlib
import jupyter_core
import pytest
import tqdm
import yaml
import warnings
import os

from torch.utils.data import DataLoader
from datasets.ucf101_dataloader import get_ucf101_datasets
from experiments.experiment import VideoActionExperiment


def run_experiments():
    os.makedirs("logs", exist_ok=True)
    sys.stdout = open("logs/experiment_runs.txt", "w")
    root_videos = "data/ucf101/raw"
    split_file_txt = "data/ucf101/splits"
    data_set_fraction = 1.0

    train_ds, val_ds = get_ucf101_datasets(
        root=root_videos,
        annotation_path=split_file_txt,
        fold=1,  # 1, 2, or 3
        frames_per_clip=8,
        fraction=data_set_fraction,
    )

    # quick sanity check
    # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    # videos, labels = next(iter(train_loader))  # ← just 2 tensors
    # print("video batch:", videos.shape)  # [B, C, T, H, W]
    # print("labels     :", labels.shape)  # [B]

    experiment = VideoActionExperiment(train_ds, val_ds, device="cuda:0")
    print("")
    experiment.train_custom(
        lr=1e-3,
        epochs=5,
        batch_size=8,
        data_set_fraction=data_set_fraction,
        freeze=True,
    )
    # experiment.run_grid_search()


def main():
    """
    Entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description="A short description of your project goes here."
    )
    parser.add_argument(
        "-r", "--run", action="store_true", help="Run the script with default settings."
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Run the script with default settings.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a configuration file (optional).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate a pretrained model without any training.",
    )
    args = parser.parse_args()
    if args.run:
        run_experiments()
    elif args.test:
        print("Installed Packages...")
        print("numpy version:", np.__version__)
        print("torch version:", torch.__version__)
        print("torchvision version:", torchvision.__version__)
        print("torchaudio version:", torchaudio.__version__)
        print("pandas version:", pandas.__version__)
        print("scipy version:", scipy.__version__)
        print("matplotlib version:", matplotlib.__version__)
        print("jupyter version:", jupyter_core.__version__)
        print("pytest version:", pytest.__version__)
        print("tqdm version:", tqdm.__version__)
        print("yaml version:", yaml.__version__)
    elif args.baseline:
        # create log folder and baseline output file
        os.makedirs("logs", exist_ok=True)
        sys.stdout = open("logs/baseline_full_resolution.txt", "w")

        # ignore warning
        warnings.filterwarnings(
            "ignore",
            message="The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.",
        )

        # Load datasets with 25% of data
        train_ds, val_ds = get_ucf101_datasets(
            root="data/ucf101/raw",
            annotation_path="data/ucf101/splits",
            fold=1,
            frames_per_clip=16,
            fraction=1,
        )
        print(f"CROP_SIZE = 224, frames_per_clip = 16")
        experiment = VideoActionExperiment(train_ds, val_ds, device="cuda:0")

        baseline_acc = experiment.run_baseline()
        print(f"Baseline Accuracy: {baseline_acc:.2%}%")

    else:
        print("No --run flag provided. Exiting...")

    # (Optional) If you have a config file:
    if args.config:
        # Add logic here to load and parse your config file.
        print(f"Using config: {args.config}")
        # For example:
        # with open(args.config, "r") as f:
        #     config_data = yaml.safe_load(f)
        #     print("Loaded configuration:", config_data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Execution interrupted by user.")
        sys.exit(1)
