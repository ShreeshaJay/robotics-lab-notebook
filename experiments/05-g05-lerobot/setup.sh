#!/usr/bin/env bash
# 05 · G0.5 in LeRobot on one NVIDIA L4: environment setup.
#
# Target VM: GCP g2-standard-8 (1x L4 24 GB, 8 vCPU, 32 GB RAM), us-central1,
# image family common-cu129-ubuntu-2404-nvidia-580 (project deeplearning-platform-release),
# 200 GB pd-balanced boot disk. See README.md.
#
# Two venvs, as LeRobot recommends (vLLM pins its own torch, so it is NOT a lerobot dependency):
#   .venv-lerobot : lerobot 0.7.0 with the g05 + libero + annotations extras (policy, sim, annotator client)
#   .venv-vllm    : vLLM server that lerobot-annotate drives over its OpenAI-compatible API
# Downloads NO model weights and NO datasets: run.py does that after the license check.
# No remote script is piped into a shell. Every pip package is pinned.
set -euo pipefail

WORK="${WORK:-$HOME/g05-lab}"
LEROBOT_VERSION="0.7.0"            # PyPI, 2026-11-05; contains src/lerobot/policies/g05
FLA_VERSION="0.6.0"                # flash-linear-attention (pure Triton): fast Gated DeltaNet path
VLLM_VERSION="0.33.0"

# lerobot>=0.7.0 ships the G0.5 integration code (src/lerobot/policies/g05) inside the wheel, and that
# code is under the G0.5 Community License, whose text says downloading counts as acceptance.
# So even this install step waits for a human decision.
if [ "${G05_LICENSE_ACK:-}" != "reviewed-g05-community-license" ]; then
  echo "Read https://huggingface.co/OpenGalaxea/G05/blob/main/licenses/LICENSE-G0.5 first, then"
  echo "  export G05_LICENSE_ACK=reviewed-g05-community-license"
  exit 1
fi

mkdir -p "$WORK"
cd "$WORK"

echo "== GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== OS packages (headless MuJoCo/EGL rendering + video encode), from the Ubuntu archive"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git ffmpeg build-essential cmake python3.12 python3.12-venv python3.12-dev \
  libegl1 libgl1 libglx0 libgles2 libglib2.0-0 libosmesa6 libxrender1 libsm6
# The Deep Learning VM driver install registers no NVIDIA EGL vendor library, so robosuite fails with
# "Cannot initialize a EGL device display". Install the GL/EGL userspace that exactly matches the running driver.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRV_MAJOR=${DRV%%.*}
GL_PKG_VER=$(apt-cache madison "libnvidia-gl-${DRV_MAJOR}-server" | awk -v v="$DRV" '$3 ~ "^"v"-" {print $3}' | sort -u | tail -1 || true)
if [ -n "$GL_PKG_VER" ]; then
  sudo apt-get install -y --no-install-recommends "libnvidia-gl-${DRV_MAJOR}-server=${GL_PKG_VER}"
else
  echo "WARNING: no libnvidia-gl-${DRV_MAJOR}-server build matching driver $DRV; EGL rendering may fail"
fi

echo "== venv 1: lerobot"
python3.12 -m venv "$WORK/.venv-lerobot"
"$WORK/.venv-lerobot/bin/python" -m pip install --upgrade "pip==25.3"
# Pin torch to the CUDA 12.8 build first (inside lerobot's torch>=2.7,<2.16 range).
"$WORK/.venv-lerobot/bin/pip" install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$WORK/.venv-lerobot/bin/pip" install "lerobot[g05,smolvla,libero,annotations]==${LEROBOT_VERSION}" \
  "flash-linear-attention==${FLA_VERSION}" \
  "robosuite==1.4.0" "bddl==1.0.1" "mujoco==3.3.1"   # same LIBERO pins as lerobot's docker/Dockerfile.sim

echo "== LIBERO config (LIBERO calls input() on import when ~/.libero/config.yaml is missing)"
LIBERO_DIR=$("$WORK/.venv-lerobot/bin/python" -c "import importlib.util, os; s=importlib.util.find_spec('libero'); print(os.path.join(os.path.dirname(s.origin), 'libero'))")
mkdir -p "$HOME/.libero" "$HOME/.cache/libero"
printf "assets: %s\nbddl_files: %s\ndatasets: %s\ninit_states: %s\n" \
  "$HOME/.libero/assets" "$LIBERO_DIR/bddl_files" "$LIBERO_DIR/../datasets" "$LIBERO_DIR/init_files" \
  > "$HOME/.libero/config.yaml"
ln -sfn "$HOME/.libero/assets" "$HOME/.cache/libero/assets"
cat "$HOME/.libero/config.yaml"

if [ "${SKIP_VLLM:-0}" != "1" ]; then   # SKIP_VLLM=1: rollouts only (no annotation), saves about 10 min
echo "== venv 2: vLLM server for lerobot-annotate"
python3.12 -m venv "$WORK/.venv-vllm"
"$WORK/.venv-vllm/bin/python" -m pip install --upgrade "pip==25.3"
"$WORK/.venv-vllm/bin/pip" install "vllm==${VLLM_VERSION}"
fi

echo "== freeze (for the write-up)"
mkdir -p "$WORK/results"
"$WORK/.venv-lerobot/bin/pip" freeze > "$WORK/results/freeze-lerobot.txt"
[ -x "$WORK/.venv-vllm/bin/pip" ] && "$WORK/.venv-vllm/bin/pip" freeze > "$WORK/results/freeze-vllm.txt"
"$WORK/.venv-lerobot/bin/python" -c "import importlib.metadata as m, torch, transformers; print('lerobot', m.version('lerobot'), 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda ok:', torch.cuda.is_available())"
"$WORK/.venv-lerobot/bin/python" -c "from lerobot.policies.g05 import configuration_g05; print('g05 policy module importable')"
[ -x "$WORK/.venv-vllm/bin/vllm" ] && "$WORK/.venv-vllm/bin/vllm" --version

df -h "$WORK"
echo "Setup done. Next: read the G0.5 Community License, then run: python3 run.py --help"
