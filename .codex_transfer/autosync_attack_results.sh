#!/usr/bin/env bash
set -u
while true; do
  rsync -a --partial -e 'ssh -o BatchMode=yes' vast-guardfed:/tmp/GuardFed/results/attack_strength/ /home/yannan/workspace/GuardFed/results/attack_strength/ >> /tmp/guardfed_attack_autosync.log 2>&1
  sleep 60
done
