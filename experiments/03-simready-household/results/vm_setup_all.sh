#!/usr/bin/env bash
# Phase 1: driver 580 -> 595 (Ubuntu signed packages, precompiled modules for running kernel) + Docker CE (official signed repo).
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
T=/var/log/scout-timings.log
echo "setup1_start $(date -u +%s)" >> $T
K=$(uname -r)
# remove the image's 580 stack first so nothing conflicts
apt-get purge -y 'nvidia-fabricmanager-580' 'libnvidia-nscq-580' || true
apt-get purge -y $(dpkg -l | awk '/^ii/ && $2 ~ /580/ && ($2 ~ /nvidia/) {print $2}') || true
apt-get install -y --no-install-recommends \
  "linux-modules-nvidia-595-server-open-$K" \
  nvidia-headless-no-dkms-595-server-open nvidia-utils-595-server \
  libnvidia-gl-595-server libnvidia-extra-595-server libnvidia-cfg1-595-server
echo "driver_installed $(date -u +%s)" >> $T
# Docker CE from Docker's signed apt repo
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
gpg --show-keys /etc/apt/keyrings/docker.asc | head -3
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io
# NVIDIA Container Toolkit already present from NVIDIA's signed repo (image); make sure it's current, wire into docker
# toolkit 1.19.1 is apt-held on the image; keep it
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
echo "docker_installed $(date -u +%s)" >> $T
apt-get install -y python3-venv
dpkg -l | awk '/^ii/ && $2 ~ /nvidia/ {print $2, $3}'
echo "setup1_done $(date -u +%s)" >> $T
mkdir -p /data/simready /data/work /data/cache/{main,computecache,logs,config,data,pkg} /data/results
cp /tmp/scene.py /tmp/run.sh /data/work/
python3 -m venv /data/venv && /data/venv/bin/pip -q install huggingface-hub==0.36.0
( s=$(date -u +%s); docker pull -q nvcr.io/nvidia/isaac-sim:6.1.0 > /data/results/docker_pull.log 2>&1; rc=$?
  echo "docker_pull rc=$rc secs=$(( $(date -u +%s) - s ))" >> $T
  docker image inspect nvcr.io/nvidia/isaac-sim:6.1.0 --format '{{.Size}} {{.Config.User}} {{json .Config.Env}}' > /data/results/image_inspect.txt 2>&1
  docker images nvcr.io/nvidia/isaac-sim >> /data/results/image_inspect.txt ) &
( cd /; s=$(date -u +%s); HF_HUB_DISABLE_TELEMETRY=1 /data/venv/bin/python -I /data/work/scene.py --download-only --assets /data/simready > /data/results/download.log 2>&1; rc=$?
  echo "asset_download rc=$rc secs=$(( $(date -u +%s) - s )) bytes=$(du -sb /data/simready | cut -f1)" >> $T ) &
wait
echo "setup_all_done $(date -u +%s)" >> $T
