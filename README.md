# 🎥 Video Action Recognition with R(2+1)D CNNs

This project applies transfer learning to video action recognition using the R(2+1)D architecture on the UCF101 dataset. It leverages pretrained Kinetics-400 weights to evaluate how well spatiotemporal features transfer to a smaller target dataset. The focus is on efficient training with frozen backbones and scaling experiments from low to high resolutions.

---

## 📚 Project Summary

A summary of the lab notes and results, including motivations, architecture decisions, preprocessing, and methodology is available here:

🔗 [View Project Summary »](./reports/lab_notes.md)

---

## 🚀 Getting Started

### 📦 1. Create the Conda Environment

Make sure you have Conda installed. Then run:

```bash
conda env create -f env.gpu.yml
conda activate video-action-env
```
## Baseline and Training Result
### 📂 2. Set Up the UCF101 Dataset

Before training, download the official UCF101.rar file and place it in data/ucf101/.

Then run:

``` bash
bash utils/setup_ucf101.sh
```
### This script will:

Create the required data/ucf101/ folder structure

Move and extract UCF101.rar

Prompt you to place the official train/test splits (trainlist01.txt, etc.) in data/ucf101/splits/

## 📊 Test Results
Metrics from our runs, including baseline evaluations and scaled experiments, can be found here:

[📈 View Test Results »](./reports/test_runs.md)

![Training Accuracy](https://github.gatech.edu/jjohns7/video_action_recognition/blob/main/src/visualization/Training_Accuracy.png)

![Training Loss](https://github.gatech.edu/jjohns7/video_action_recognition/blob/main/src/visualization/Training_loss.png)

![Training and Testing Accuracy](https://github.gatech.edu/jjohns7/video_action_recognition/blob/main/src/visualization/Training_Testing_Accuracy.png)





