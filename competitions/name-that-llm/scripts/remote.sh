#!/usr/bin/env bash
# Run CPU jobs on the hoody host (12 cores, 6 GB RAM, no GPU). Usage:
#   scripts/remote.sh sync                 # rsync code + data + oof to hoody:~/ntl (never tmp/ or kernel outputs)
#   scripts/remote.sh run <name> <cmd...>  # sync, then nohup the command in ~/ntl, log to ~/ntl/logs/<name>.log
#   scripts/remote.sh log <name>           # tail the log
#   scripts/remote.sh pull                 # copy hoody's oof/ and blend/ back here
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
H=hoody; R='~/ntl'
case "${1:-}" in
  sync) rsync -az --exclude 'tmp/' --exclude 'kernels/*/out' --exclude 'kernels/*/models' --exclude '__pycache__' --exclude 'dist/' -e "ssh -o BatchMode=yes" "$HERE/" "$H:$R/" ;;
  run) name="$2"; shift 2; "$0" sync; ssh -o BatchMode=yes "$H" "cd $R && mkdir -p logs && setsid nohup .venv/bin/python $* > logs/$name.log 2>&1 < /dev/null & echo started" ;;
  log) ssh -o BatchMode=yes "$H" "tail -n 40 $R/logs/$2.log" ;;
  pull) rsync -az -e "ssh -o BatchMode=yes" "$H:$R/oof/" "$HERE/oof/"; rsync -az -e "ssh -o BatchMode=yes" "$H:$R/logs/" "$HERE/tmp/hoody_logs/" ;;
  *) echo "usage: $0 sync|run <name> <cmd>|log <name>|pull"; exit 1 ;;
esac
