# Dockerfile for Video Action Recognition (GPU + conda + pip-based PyTorch)

# 1) Base image: NVIDIA CUDA 11.8, Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2) Install essential packages
RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

# 3) Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 4) Copy env file & create the conda environment
# Make sure your 'env.gpu.yml' is named accordingly and references the environment name you want
COPY env.gpu.yml /tmp/env.gpu.yml
RUN conda env create -f /tmp/env.gpu.yml && conda clean -afy

# 5) Activate the environment by default
# If 'env.gpu.yml' sets 'name: video_action_recognition_env', do this:
ENV PATH=/opt/conda/envs/video_action_recognition_gpu/bin:$PATH
ENV CONDA_DEFAULT_ENV=video_action_recognition_gpu

# 6) Setup workspace
WORKDIR /workspace
COPY . /workspace

# 7) Default command: open bash
CMD ["/bin/bash"]
