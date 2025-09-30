#!/bin/bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to provision the Rex environment. Please install Miniconda or Anaconda first." >&2
  exit 1
fi

TARGET="${1:-cpu}"

conda create -n rex python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rex

if [[ "${TARGET}" == "gpu" ]]; then
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
  conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

pip install -r requirements.txt
