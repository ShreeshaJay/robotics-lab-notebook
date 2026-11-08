#!/usr/bin/env bash
# usage: run.sh <tag> [scene.py args...]   e.g. run.sh convexhull   |  run.sh sdf --collider sdf
# EULA: I accepted the NVIDIA Omniverse License Agreement for Isaac Sim 6.1 on 2026-11-08.
# For that run the runner exported ACCEPT_EULA=Y on that VM only; this copy has no default. PRIVACY_CONSENT deliberately not set.
set -uo pipefail
TAG=$1; shift
OUT=/data/out_$TAG; mkdir -p $OUT /data/cache/kit; chown -R 1234:1234 /data/cache $OUT 2>/dev/null || true
U=$(docker image inspect nvcr.io/nvidia/isaac-sim:6.1.0 --format '{{.Config.User}}'); U=${U:-1234:1234}
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw --format=csv,nounits -lms 2000 > $OUT/gpu_${TAG}.csv &
SMI=$!
s=$(date -u +%s)
docker rm -f scout-$TAG >/dev/null 2>&1; docker run --name scout-$TAG --gpus all -e ACCEPT_EULA="${ACCEPT_EULA:?read and accept the NVIDIA Isaac Sim EULA yourself, then export ACCEPT_EULA=Y}" -u "$U" \
  -v /data/cache/main:/isaac-sim/.cache:rw -v /data/cache/kit:/isaac-sim/kit/cache:rw -v /data/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v /data/cache/logs:/isaac-sim/.nvidia-omniverse/logs:rw -v /data/cache/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v /data/cache/data:/isaac-sim/.local/share/ov/data:rw -v /data/cache/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v /data/simready:/assets:ro -v /data/work:/work:ro -v $OUT:/out:rw \
  --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.1.0 /work/scene.py --assets /assets --out /out "$@" \
  > $OUT/run_${TAG}.log 2>&1
rc=$?; docker diff scout-$TAG 2>/dev/null | awk "{print $2}" | cut -d/ -f1-5 | sort | uniq -c | sort -rn | head -25 > $OUT/docker_diff_${TAG}.txt; docker rm scout-$TAG >/dev/null
kill $SMI
echo "run_$TAG rc=$rc secs=$(( $(date -u +%s) - s ))" | tee -a /var/log/scout-timings.log
