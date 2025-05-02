# Project Lab Notes: Video Action Recognition with 3D CNNs

This document summarizes our team’s practical exploration into using transfer learning for video action recognition tasks, specifically applying a pretrained R(2+1)D neural network to the UCF101 dataset. We started by utilizing pretrained weights from Kinetics-400, aiming to determine whether simply freezing the backbone and adjusting only the final classifier can effectively capture spatiotemporal patterns on a new dataset.

---

## 1. Introduction and Motivation

### 1.1 Why this Approach Matters

Video action recognition can be challenging due to the added complexity introduced by the temporal dimension. Transfer learning helps alleviate this challenge by leveraging knowledge learned from large-scale datasets. We selected the R(2+1)D architecture because of its unique structure, which separates spatial and temporal convolutions, allowing it to efficiently learn richer, more meaningful features than standard 3D CNNs.

### 1.2 Why Kinetics-400?

The pretrained weights from Kinetics-400 offer a rich representation of diverse human actions. Although UCF101 is smaller and less diverse, our hypothesis is that these robust, pretrained features will generalize well to the new action classes.

---

## 2. Our Goals

- **Validate Transfer Learning**: Confirm whether freezing the backbone and fine-tuning just the classifier head is sufficient for effective action classification on UCF101.
- **Efficiency First**: To quickly iterate and handle limited computing resources, we initially chose lower resolutions (112×112 pixels) and fewer frames per clip (8 frames).
- **Gradually Scale**: After initial success, we planned to increase the resolution and the number of frames (224×224 resolution, 16 frames per clip) to see if accuracy improved significantly.

---

## 3. Experimental Setup

### 3.1 R(2+1)D Architecture at a Glance

The R(2+1)D model uniquely splits convolutions:

- **Spatial Convolutions**: Extract visual details from individual frames.
- **Temporal Convolutions**: Identify motion patterns across multiple frames.

This structure provides efficiency while effectively capturing detailed spatiotemporal information.

### 3.2 Transfer Learning Strategy

- Replaced the final fully connected layer (originally 400 classes) to classify the 101 action classes of UCF101.
- Froze all other layers to retain pretrained knowledge and streamline training.

### 3.3 Data Handling

#### Preprocessing
- **Frame Sampling**: Initially 8 frames, later scaled to 16.
- **Tensor Conversion**: Reordered dimensions `[T,H,W,C]` → `[C,T,H,W]` to match PyTorch conventions.
- **Spatial Cropping**: Initially set resolution at 112×112, later increasing to 224×224.
- **Normalization**: Adjusted input data using Kinetics-400 statistics to ensure compatibility with pretrained parameters.

#### Data Loader
- Avoided lambda functions in preprocessing due to Windows multiprocessing compatibility issues.
- Temporarily reduced dataset size during initial tests for faster experimentation.

---

## 4. Overcoming Challenges

### 4.1 Computational Limits
- **Challenge**: High-resolution inputs (224×224) and more frames increased computational demands.
- **Our Solution**: Began with smaller-scale experiments (112×112 resolution, 8 frames) for quick initial results.

### 4.2 Technical Issues with Windows
- **Issue**: Lambda functions caused multiprocessing errors on Windows.
- **Solution**: Replaced them with globally defined functions or `functools.partial` to resolve compatibility.

### 4.3 MLflow Paths
- **Issue**: Windows-specific formatting issues in MLflow URIs (`file://` paths).
- **Solution**: Adjusted paths to standard Windows paths or correctly formatted URIs.

---

## 5. Evaluation and Next Steps

### 5.1 Evaluation Metrics
- Main focus on top-1 and top-5 accuracy.
- Also evaluating class-level precision, recall, and confusion matrices for deeper insights.

### 5.2 Qualitative Checks
- Reviewing predictions and activations visually to understand and validate the model's decision-making process.

### 5.3 Scaling Experiments
- Planning further tests at higher resolution and increased temporal frames, with optimized hyperparameters like learning rate (e.g., 1e-3, 1e-4).
- Aiming to measure the exact accuracy improvements from higher spatial and temporal resolutions alone.

---

## 6. Final Thoughts

Our structured use of transfer learning and incremental scaling demonstrates an efficient path to high-quality video action recognition. Initial low-resolution tests allowed fast experimentation, setting a clear foundation for later, higher-resolution analyses. Moving forward, we aim to clearly quantify the benefits of increased resolution and frame counts, deepening our understanding of transfer learning’s practical effectiveness in video classification tasks.

