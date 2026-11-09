# Time-capped GPU VM

`create_vm.sh` creates one GCP VM with a single NVIDIA L4 for a short experiment. Three safeguards:
- **Hard cap:** `--max-run-duration` deletes the VM after `HOURS` (default 4, max 8 without editing the script).
- **Idle power-off:** a startup script shuts the VM down after `IDLE_MIN` minutes (default 30) of near-zero GPU and CPU use.
- **Labels:** `owner`, `purpose` and `exp` labels make the spend easy to find in billing.

Copy results off the VM before the cap hits (`gcloud compute scp`), because the disk is deleted with the VM.

With `ON_END=STOP` (for Spot runs), a preemption or the time cap stops the VM and keeps its disk, so setup survives;
restart it with `gcloud compute instances start`. The stopped disk is still billed, so delete the VM yourself at the end.
On `a2-ultragpu` machines the bundled local SSD is discarded at the stop.
