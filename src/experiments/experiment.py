# experiments/experiment.py
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from sklearn.model_selection import GridSearchCV

from models.r2plus1d_net import R2Plus1DNet
from utils.dataset_wrappers import XYDataset
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import mlflow
import mlflow.pytorch
import os


class VideoActionExperiment:
    def __init__(self, train_dataset, val_dataset, device="cuda:0"):
        self.train_dataset = XYDataset(train_dataset)
        self.val_dataset = XYDataset(val_dataset)
        actual_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = actual_device

        if actual_device.type == "cuda":
            # Print which GPU name we’re using
            gpu_index = actual_device.index if actual_device.index is not None else 0
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            # CPU mode
            print("Using CPU (no CUDA GPU available)")

    def get_skorch_net(self):
        net = NeuralNetClassifier(
            module=R2Plus1DNet,
            module__num_classes=101,
            module__pretrained=True,
            module__freeze=True,
            lr=1e-3,
            max_epochs=5,
            device=self.device,
            train_split=predefined_split(self.val_dataset),
        )
        return net

    def run_baseline(self, batch_size: int = 8, num_workers: int = 8):
        print(
            "Running baseline evaluation (pretrained backbone + random UCF101 head)..."
        )

        # Load backbone pretrained on Kinetics-400, but random-initialize 101-class head
        model = R2Plus1DNet(num_classes=101, pretrained=True, freeze=True).to(
            self.device
        )
        model.eval()

        for param in model.backbone.fc.parameters():
            param.requires_grad = False

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        correct, total = 0, 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = model(videos)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        baseline_accuracy = correct / total
        print(f"Baseline accuracy on validation set: {baseline_accuracy:.3%}%")

        return baseline_accuracy

    def run_grid_search(self):
        net = self.get_skorch_net()
        param_grid = {
            "lr": [1e-3, 1e-4],
            "max_epochs": [5, 10],
        }
        gs = GridSearchCV(net, param_grid, scoring="accuracy", refit=True, cv=3)
        gs.fit(X=None, y=None)  # Skorch uses the dataset inside

        print("Best score:", gs.best_score_)
        print("Best params:", gs.best_params_)
        return gs.best_estimator_

    def train_custom(
        self,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 8,
        num_workers: int = 8,
        data_set_fraction: float = 0.25,
        freeze: bool = True,
    ):
        model = R2Plus1DNet(101, pretrained=True, freeze=freeze).to(self.device)

        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        crit = nn.CrossEntropyLoss()

        train_ld = DataLoader(
            self.train_dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_ld = DataLoader(
            self.val_dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        history = {"epoch": [], "train_loss": [], "val_acc": []}
        best_acc = 0.0

        print("Starting Training with MLflow Tracking:")
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        mlflow.set_experiment("video_action_classification")

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "freeze_pretrained": freeze,
                    "dataset_fraction": data_set_fraction,
                }
            )

            for ep in range(1, epochs + 1):
                # training
                start_epoch_time = time.time()  # Start the timer
                print(f"Starting epoch {ep} ...")
                model.train()
                running = 0.0
                for videos, labels in train_ld:  # already 2‑tuple
                    videos, labels = videos.to(self.device), labels.to(self.device)
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model(videos), labels)
                    loss.backward()
                    opt.step()
                    running += loss.item() * videos.size(0)
                train_loss = running / len(train_ld.dataset)

                # validation
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for videos, labels in val_ld:
                        videos, labels = videos.to(self.device), labels.to(self.device)
                        preds = model(videos).argmax(1)
                        correct += (preds == labels).sum().item()
                        total += labels.numel()
                val_acc = correct / total

                # Record how long this epoch took
                epoch_time = time.time() - start_epoch_time
                # store info
                history["epoch"].append(ep)
                history["train_loss"].append(train_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {ep:02}, train_loss={train_loss:.4f}, val_acc={val_acc:.3%} epoch_time={epoch_time:.2f} sec"
                )
            # Log best model
            best_acc = max(history["val_acc"])
            mlflow.log_metric("best_val_accuracy", best_acc)
            print(f"Best Validation Accuracy {best_acc:.3%}")
            # Log model artifact
            mlflow.pytorch.log_model(model, "model")

        return model, best_acc, history
