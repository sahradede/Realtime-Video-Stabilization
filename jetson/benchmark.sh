#!/usr/bin/env bash
# Run the same clips the notebook uses, on the device.
#
#   ./benchmark.sh                    # benchmark clips only
#   ./benchmark.sh my_recording.mkv   # your own clip too
#
# The benchmark clips are 640x360 and 876x446, so they will look fast. The
# number that decides whether this fits a 30 FPS budget is your own 1080p
# footage -- pass it as an argument.

set -e
FRAMES=${FRAMES:-300}

fetch() {                      # fetch <file> <url>
    [ -f "$1" ] && { echo "[cached] $1"; return; }
    echo "[downloading] $1"
    wget -q -O "$1" "$2" || { echo "  failed, skipping"; rm -f "$1"; }
}

fetch meshflow.avi \
    https://raw.githubusercontent.com/sudheerachary/Mesh-Flow-Video-Stabilization/master/data/shaky-5.avi
fetch ostrich.mp4 \
    https://s3.amazonaws.com/python-vidstab/ostrich.mp4

CLIPS="meshflow.avi ostrich.mp4"
[ -f running.mp4 ] && CLIPS="$CLIPS running.mp4"
CLIPS="$CLIPS $*"

echo
echo "budget for 30 FPS is 33.3 ms/frame -- watch p95, not the mean"
echo

for f in $CLIPS; do
    [ -f "$f" ] || { echo "skipping $f (not found)"; continue; }
    echo "=== $f ==="
    python3 stabilize.py "$f" --frames "$FRAMES"
    echo
done

echo "If p95 is over budget, try:"
echo "  python3 stabilize.py <clip> --frames $FRAMES --proc-width 320"
echo "  python3 stabilize.py <clip> --frames $FRAMES --proc-width 320 --crop 0.06"
