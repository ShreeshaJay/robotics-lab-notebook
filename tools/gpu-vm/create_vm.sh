#!/usr/bin/env bash
# Create one time-capped GPU VM on GCP for a short experiment.
# Safeguards: hard max run duration (VM is DELETED when it expires, or STOPPED with ON_END=STOP),
# idle shutdown (GPU idle for IDLE_MIN minutes -> power off), and labels for cost tracking.
# Usage: PROJECT=my-project NAME=exp-01 [HOURS=4] [ZONE=us-central1-a] [SPOT=1] [MACHINE=a2-ultragpu-1g] ./create_vm.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"; : "${NAME:?set NAME}"
ZONE="${ZONE:-us-central1-a}"
HOURS="${HOURS:-4}"           # hard cap; VM is deleted after this
IDLE_MIN="${IDLE_MIN:-30}"
ON_END="${ON_END:-DELETE}"     # DELETE (default) or STOP: STOP keeps the disk through Spot preemptions so setup survives; you must delete it yourself
MACHINE="${MACHINE:-g2-standard-8}"   # g2-standard-8: 1x L4 24 GB; a2-ultragpu-1g: 1x A100 80 GB (GPU is bundled with both)
DISK_GB="${DISK_GB:-200}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

if [ "$HOURS" -gt 8 ]; then echo "HOURS > 8 needs explicit approval"; exit 1; fi
case "$ON_END" in DELETE|STOP) ;; *) echo "ON_END must be DELETE or STOP"; exit 1;; esac

PROVISIONING=()
if [ "${SPOT:-0}" = "1" ]; then PROVISIONING=(--provisioning-model=SPOT); fi

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --maintenance-policy=TERMINATE \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
  --max-run-duration="${HOURS}h" --instance-termination-action="$ON_END" \
  "${PROVISIONING[@]}" \
  --labels=owner=tech-scout,purpose=experiment,exp="$NAME" \
  --metadata=install-nvidia-driver=True,idle-minutes="$IDLE_MIN" \
  --metadata-from-file=startup-script="$(dirname "$0")/idle_shutdown.sh"

echo "Created $NAME. Hard stop ($ON_END) in ${HOURS}h; idle power-off after ${IDLE_MIN} min."
if [ "$ON_END" = "STOP" ]; then echo "WARNING: ON_END=STOP leaves a stopped VM and its disk (still billed). Restart after preemption with: gcloud compute instances start $NAME --project=$PROJECT --zone=$ZONE"; fi
echo "Delete when done:  gcloud compute instances delete $NAME --project=$PROJECT --zone=$ZONE --quiet"
