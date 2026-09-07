#!/usr/bin/env python3
"""
Stabilize a video file. Input in, stabilized video out, nothing else.

    python3 stabilize.py input.mp4
    python3 stabilize.py input.mp4 -o output.mp4
    python3 stabilize.py input.mp4 --side-by-side      # raw | stabilized
    python3 stabilize.py input.mkv --crop 0.06         # keep more of the frame

Processing is frame-by-frame with no look-ahead, exactly as it would run on a
live stream -- reading from a file changes nothing about the result.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from core import StabConfig, OnlineStabilizer, VideoFileSource


def main():
    p = argparse.ArgumentParser(description="Stabilize a video file")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None,
                   help="default: <input>_stab.mp4")
    p.add_argument("--alpha", type=float, default=0.92,
                   help="smoothing, 0.85 responsive .. 0.97 very smooth")
    p.add_argument("--crop", type=float, default=0.08,
                   help="crop per side; 0.08 loses 16%% of the frame")
    p.add_argument("--max-angle", type=float, default=3.0)
    p.add_argument("--proc-width", type=int, default=480,
                   help="motion-estimation width; lower is faster")
    p.add_argument("--side-by-side", action="store_true",
                   help="write raw and stabilized next to each other")
    p.add_argument("--frames", type=int, default=0, help="0 = whole clip")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    if not os.path.exists(a.input):
        print(f"not found: {a.input}", file=sys.stderr)
        return 1

    out_path = a.output or (os.path.splitext(a.input)[0] + "_stab.mp4")
    cfg = StabConfig(smooth_alpha=a.alpha, crop_ratio=a.crop,
                     max_angle_deg=a.max_angle, proc_width=a.proc_width,
                     adaptive=False)

    src = VideoFileSource(a.input)
    stab = OnlineStabilizer(cfg)
    W, H = src.width, src.height
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             src.fps, (W*2 if a.side_by_side else W, H))
    if not writer.isOpened():
        print(f"could not open writer for {out_path}", file=sys.stderr)
        return 1

    if not a.quiet:
        print(f"{W}x{H} @ {src.fps:.1f} fps  |  alpha={cfg.smooth_alpha} "
              f"crop={cfg.crop_ratio} (zoom {1/(1-2*cfg.crop_ratio):.2f}x)")

    times, n = [], 0
    t0 = time.perf_counter()
    while True:
        f = src.read()
        if f is None:
            break
        sf = stab.process(f)
        times.append(sf.proc_ms)
        writer.write(np.hstack([f.image, sf.image]) if a.side_by_side
                     else sf.image)
        n += 1
        if not a.quiet and n % 100 == 0:
            print(f"  {n} frames", end="\r", flush=True)
        if a.frames and n >= a.frames:
            break
    writer.release()
    src.release()

    if not n:
        print("no frames processed", file=sys.stderr)
        return 1

    t = np.array(times)
    if not a.quiet:
        print(f"\n{n} frames in {time.perf_counter()-t0:.1f}s")
        print(f"{t.mean():.2f} ms/frame (p95 {np.percentile(t,95):.2f}) "
              f"-> {1000/t.mean():.1f} FPS")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
