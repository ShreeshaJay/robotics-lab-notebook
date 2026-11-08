#!/usr/bin/env bash
# 04 · Flora on one NVIDIA L4: environment setup.
#
# Target VM: GCP g2-standard-8 (1x L4 24 GB, 8 vCPU, 32 GB RAM), us-central1,
# image family common-cu129-ubuntu-2404-nvidia-580 (project deeplearning-platform-release),
# 400 GB pd-ssd boot disk. See README.md.
#
# What this does: installs OS packages from the Ubuntu archive, clones Flora at a pinned
# commit, and builds the two Python 3.12 venvs the Flora README asks for (they pin different
# PyTorch builds and must not be mixed). It downloads NO model weights and NO video: run.py
# does that, after you confirm the license.
#
# No remote script is piped into a shell. Every pip package is pinned.
set -euo pipefail

WORK="${WORK:-$HOME/flora-lab}"
FLORA_COMMIT="d8b755ec9a990418b1f31c7f48c75341c7229ea6"   # NVlabs/Flora main, 2026-11-05
BNB_VERSION="0.50.2"                                       # only for the experimental 4-bit L4 path
WITH_BNB="${WITH_BNB:-1}"                                  # set WITH_BNB=0 on an 80 GB GPU (full precision, no bitsandbytes)

mkdir -p "$WORK"
cd "$WORK"

echo "== GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== OS packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends git ffmpeg python3.12 python3.12-venv python3.12-dev build-essential

echo "== Flora code (pinned commit)"
if [ ! -d Flora/.git ]; then
  git clone https://github.com/NVlabs/Flora.git Flora
fi
git -C Flora fetch --depth 1 origin "$FLORA_COMMIT"
git -C Flora checkout --detach "$FLORA_COMMIT"
git -C Flora log -1 --format='Flora at %H (%cd)'

echo "== venv 1: annotation (TempSeg -> Captioner), pins from captioner/batch_utils/requirements.txt"
python3.12 -m venv "$WORK/.venv-annotation"
"$WORK/.venv-annotation/bin/python" -m pip install --upgrade "pip==25.3"
"$WORK/.venv-annotation/bin/pip" install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$WORK/.venv-annotation/bin/pip" install -r Flora/captioner/batch_utils/requirements.txt
if [ "$WITH_BNB" = "1" ]; then "$WORK/.venv-annotation/bin/pip" install "bitsandbytes==${BNB_VERSION}"; fi

echo "== venv 2: rewarder (vLLM), pins from rewarder/batch_utils/requirements.txt"
python3.12 -m venv "$WORK/.venv-rewarder"
"$WORK/.venv-rewarder/bin/python" -m pip install --upgrade "pip==25.3"
"$WORK/.venv-rewarder/bin/pip" install -r Flora/rewarder/batch_utils/requirements.txt
if [ "$WITH_BNB" = "1" ]; then "$WORK/.venv-rewarder/bin/pip" install "bitsandbytes==${BNB_VERSION}"; fi

echo "== freeze (for the write-up)"
mkdir -p "$WORK/results"
"$WORK/.venv-annotation/bin/pip" freeze > "$WORK/results/freeze-annotation.txt"
"$WORK/.venv-rewarder/bin/pip" freeze > "$WORK/results/freeze-rewarder.txt"
"$WORK/.venv-annotation/bin/python" -c "import torch; print('annotation torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"
"$WORK/.venv-rewarder/bin/python" -c "import torch, vllm; print('rewarder torch', torch.__version__, 'vllm', vllm.__version__, 'cuda ok:', torch.cuda.is_available())"

df -h "$WORK"
echo "Setup done. Next: read the NVIDIA License (Flora/LICENSE), then run: python3 run.py --help"
