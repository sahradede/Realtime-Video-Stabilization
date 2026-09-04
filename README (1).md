# Keel

Causal video stabilization — no look-ahead, no added latency. Built for live
streaming on a Jetson Orin Nano, with a synchronized stereo pair as the next step.

## Why causal

Most stabilization code smooths the whole camera trajectory with a sliding
window, which needs future frames. That's fine offline and impossible live.
Here the trajectory is smoothed with an IIR filter instead, so frame *i* is
produced from frames 0..*i* only and comes out immediately.

The cost of that constraint was measured rather than assumed: a look-ahead
variant with up to 400 ms of buffering was tested and did not beat the causal
version on these clips.

## How it works

```
frame -> downscale + gray -> Shi-Tomasi + Lucas-Kanade
      -> partial affine (RANSAC) -> cumulative trajectory
      -> EMA smoothing -> correction -> corner-checked limit
      -> single warpAffine (rotation + translation + crop-zoom)
```

Motion is estimated on a downscaled grayscale frame; the warp runs at full
resolution. Rotation, translation and the crop-zoom go into one matrix.

## Results

Three clips, one config (`alpha=0.92`, `crop_ratio=0.08`, 16% of the frame):

| clip | translation | rotation | frames at crop limit |
|---|---|---|---|
| meshflow (MeshFlow ECCV 2016 sample) | 64% | 69% | 0% |
| ostrich (handheld) | 36% | n/a¹ | 4% |
| running (fast forward motion) | 25% | 42% | 6% |

Ground truth test on meshflow — inject known shake, measure what comes back
out: **x 87%, y 84%, angle 66%**.

1080p on a shared Colab CPU: **17 ms/frame, p95 27 ms**.

¹ That clip has 0.075° of rotation to begin with; the ratio is noise, not signal.

## Known limits

**Global affine model.** One transform is applied to the whole frame, so
rolling shutter and parallax are out of scope by construction. A quadrant test
makes this concrete: split the frame into four and measure each separately, and
on handheld footage the quadrants disagree more than they agree (131% of the
common motion). No single affine transform can fix that — correcting one corner
breaks another. Getting past it needs a spatially-varying model like MeshFlow.

**Crop budget.** `crop_ratio` is field of view traded for correction range. The
curve knees around 0.08; beyond that each extra 4% of frame buys roughly 2
points. Clips with sustained fast motion (running, driving) saturate the budget
tracking the intentional motion and leave less for the shake.

**Metric caveats.** End-to-end suppression includes the deliberate camera
motion, which is supposed to survive, so it can't approach 100% on pan-heavy
footage. High-frequency suppression (`hf_score`) is the number to tune against.

## Layout

| file | role |
|---|---|
| `core.py` | the stabilizer — the only file the Jetson needs |
| `tools.py` | measurement, benchmarks, ground truth |
| `plots.py`, `runner.py` | notebook helpers |
| `run_camera.py` | live capture + performance reporting |
| `record_clips.py` | raw recording from CSI cameras |
| `keel.ipynb` | full run plus the appendix of experiments |

## Jetson

JetPack ships a CUDA-enabled OpenCV. Don't `pip install opencv-python`, it
replaces it with a build that has no CUDA.

```bash
gst-launch-1.0 nvarguscamerasrc num-buffers=30 ! fakesink   # camera alive?
python3 record_clips.py --sensor-id 0 --seconds 30          # real footage
sudo nvpmodel -m 0 && sudo jetson_clocks
python3 run_camera.py --sensor-id 0 --frames 900 --no-display
```

Tune on your own recording, not on the benchmarks — vibration profiles differ
enough that parameters don't transfer.

Watch p95, not the mean. 30 FPS means every frame under 33.3 ms; a 20 ms mean
with a 40 ms p95 still drops frames. Close VS Code while measuring, its server
takes a real bite out of an Orin Nano.

If p95 is over budget, in order: `--proc-width 320`, then `--cuda` for the warp,
then VPI for optical flow.

## Stereo

Stabilizing two cameras independently breaks the stereo geometry — disparity
and epipolar alignment drift apart. `process()` takes an `external_correction`
and returns the correction it applied, so the sync layer can compute one shared
correction and force both cameras to use it.

Still open: hardware timestamps. `CsiCameraSource` tries `CAP_PROP_POS_MSEC`
and falls back to the wall clock, which includes buffer delay. A few ms of skew
is enough to ruin pairing, so the real PTS has to come off appsink before sync
work starts.

## Credits

Benchmark clips from the [MeshFlow](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization)
repository (Liu et al., ECCV 2016) and [python_video_stab](https://github.com/AdamSpannbauer/python_video_stab).
