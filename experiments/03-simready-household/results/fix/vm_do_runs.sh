#!/usr/bin/env bash
# Runner used on the 2026-11-08 fix VM. ACCEPT_EULA is NOT set here: export it yourself after reading the EULA (run.sh refuses otherwise). PRIVACY_CONSENT stays unset.
cd /data/work
for spec in "$@"; do
  tag=${spec%%:*}; extra=${spec#*:}; [ "$extra" = "$spec" ] && extra=""
  bash /data/work/run.sh "$tag" --frames 2 $extra
done
echo ALL_RUNS_DONE $(date -u +%s) >> /var/log/scout-timings.log
