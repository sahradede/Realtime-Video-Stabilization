# Stabilization — file-based test

Two files, no camera needed. Point it at a video and it writes a stabilized one.

| file | role |
|---|---|
| `core.py` | the stabilizer (library, not run directly) |
| `stabilize.py` | command-line script |

## Setup

Check what's installed:

```bash
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

Do **not** run `pip install opencv-python` on a Jetson — JetPack already ships a
CUDA-enabled OpenCV and pip would replace it with a build that has none.

## Run

```bash
python3 stabilize.py clip.mp4
```

Writes `clip_stab.mp4` and prints timing:

```
1920x1080 @ 30.0 fps  |  alpha=0.92 crop=0.08 (zoom 1.19x)
300 frames in 8.4s
28.10 ms/frame (p95 41.20) -> 35.6 FPS
```

Side by side, raw on the left:

```bash
python3 stabilize.py clip.mp4 --side-by-side
```

Just the first 300 frames, for a quick look:

```bash
python3 stabilize.py clip.mp4 --frames 300
```

## Options

| flag | default | what it does |
|---|---|---|
| `-o` | `<input>_stab.mp4` | output path |
| `--alpha` | 0.92 | smoothing; 0.85 responsive, 0.97 very smooth |
| `--crop` | 0.08 | crop per side; 0.08 costs 16% of the frame |
| `--proc-width` | 480 | motion-estimation width; lower is faster |
| `--max-angle` | 3.0 | rotation limit in degrees |
| `--frames` | 0 (all) | stop after N frames |
| `--side-by-side` | off | write raw and stabilized together |

## Reading the timing

Watch **p95, not the mean**. 30 FPS means every frame under 33.3 ms; a 20 ms
average with a 40 ms p95 still drops frames.

Close VS Code before measuring and run from an SSH terminal — the VS Code server
and Python language server take a real bite out of an Orin Nano's CPU, and the
numbers move around if you don't.

If p95 is over budget, try these and record the difference:

```bash
python3 stabilize.py clip.mp4 --frames 300 --proc-width 320
python3 stabilize.py clip.mp4 --frames 300 --proc-width 320 --crop 0.06
```

Next steps beyond that are moving the warp to the GPU (`cv2.cuda.warpAffine`)
and optical flow to VPI, which uses the PVA/VIC hardware blocks.

## Notes

`.mkv` files sometimes report a bogus frame count through OpenCV. It doesn't
affect processing, but if something looks off, either use `--frames` to bound it
or remux first:

```bash
ffmpeg -i clip.mkv -c copy clip.mp4
```

Processing is frame-by-frame with no look-ahead, exactly as it would run on a
live stream — reading from a file changes nothing about the result.
