#!/usr/bin/env bash
# Startup script: power off the VM when the GPU and CPU have been idle for N minutes.
IDLE_MIN=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/idle-minutes" || echo 30)
cat > /usr/local/bin/idle-watch.sh <<WATCH
#!/usr/bin/env bash
idle=0
while true; do
  sleep 60
  gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  gpu=\${gpu:-0}
  load=\$(awk '{print int(\$1*100)}' /proc/loadavg)
  if [ "\$gpu" -lt 5 ] && [ "\$load" -lt 50 ]; then idle=\$((idle+1)); else idle=0; fi
  if [ "\$idle" -ge $IDLE_MIN ]; then logger "idle-watch: idle \$idle min, powering off"; shutdown -h now; fi
done
WATCH
chmod +x /usr/local/bin/idle-watch.sh
nohup /usr/local/bin/idle-watch.sh >/var/log/idle-watch.log 2>&1 &
