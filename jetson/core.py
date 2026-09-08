# Core stabilizer: corner-based budget, soft limiting,
# grid-distributed features, confidence weighting.
"""
Causal video stabilization for live streams.

Design notes
------------
1) No future frames. process() only ever sees what has already arrived.
   Trajectory smoothing is an EMA (IIR), not a sliding window, so the
   output frame is produced immediately with zero added latency.
2) Sources are pluggable. VideoFileSource for offline work today,
   CsiCameraSource on the Jetson tomorrow; nothing else changes.
3) Built with stereo in mind. Every frame carries a timestamp, a camera
   id and the warp that was applied. process() also accepts an
   external_correction so a future sync layer can force both cameras to
   use the *same* correction -- stabilizing two cameras independently
   would break the stereo geometry.
4) Speed. Motion is estimated on a downscaled grayscale frame; rotation,
   translation and the crop-zoom all go into a single warpAffine.
"""
from __future__ import annotations
import math, os, time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, List
import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    index: int
    cam_id: int = 0


@dataclass
class Motion:
    dx: float = 0.0
    dy: float = 0.0
    da: float = 0.0          # radians
    valid: bool = False
    n_inliers: int = 0
    def as_array(self):
        return np.array([self.dx, self.dy, self.da], dtype=np.float64)


@dataclass
class StabilizedFrame:
    image: np.ndarray
    timestamp: float
    index: int
    cam_id: int
    correction: np.ndarray
    motion: Motion
    warp: np.ndarray
    n_tracks: int
    proc_ms: float


@dataclass
class StabConfig:
    proc_width: int = 480          # motion-estimation width; biggest speed lever
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 12
    block_size: int = 3
    redetect_interval: int = 15
    min_tracks: int = 60
    lk_win: int = 21
    lk_levels: int = 3
    use_fb_check: bool = True      # forward-backward check (~20% slower)
    fb_threshold: float = 1.0
    ransac_thresh: float = 3.0
    min_inliers: int = 12
    # Defaults come from the benchmark sweep (see the appendix notebook).
    # adaptive=True was measured to fire almost constantly and wreck the
    # smoothing, so it is off by default. It may still help on footage with
    # fast sustained panning -- check visually before turning it back on.
    smooth_alpha: float = 0.90     # 0.85 = responsive, 0.97 = very smooth
    adaptive: bool = False
    crop_ratio: float = 0.12       # crop per side; zoom = 1/(1-2r)
    grid: int = 4                  # feature distribution grid
    conf_inliers: int = 40         # inlier count for full confidence
    windup_leak: float = 0.25      # anti-windup leak rate
    max_angle_deg: float = 3.0
    safety: float = 0.95           # unused since the corner check replaced it
    num_threads: int = 0
    interpolation: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_REPLICATE


class FrameSource:
    def read(self) -> Optional[Frame]:
        raise NotImplementedError
    def release(self): pass
    def __iter__(self):
        while True:
            f = self.read()
            if f is None: break
            yield f


class VideoFileSource(FrameSource):
    """Reads a file but behaves like a camera: no seeking, no look-ahead."""
    def __init__(self, path, cam_id=0, realtime=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dosya bulunamadi: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Video acilamadi (codec?): {path}")
        self.cam_id, self.realtime, self.index = cam_id, realtime, -1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._t0 = time.monotonic()

    def read(self):
        ok, img = self.cap.read()
        if not ok: return None
        self.index += 1
        ts = self.index / self.fps
        if self.realtime:
            d = (self._t0 + ts) - time.monotonic()
            if d > 0: time.sleep(d)
        return Frame(img, ts, self.index, self.cam_id)

    def release(self): self.cap.release()


class CsiCameraSource(FrameSource):
    """Jetson CSI camera. Won't run off-device; here so deployment is a no-op."""
    GST = ("nvarguscamerasrc sensor-id={sid} ! "
           "video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
           "nvvidconv ! video/x-raw, format=BGRx ! "
           "videoconvert ! video/x-raw, format=BGR ! "
           "appsink drop=true max-buffers=1 sync=false")

    def __init__(self, sensor_id=0, width=1920, height=1080, fps=30, cam_id=None):
        self.cap = cv2.VideoCapture(
            self.GST.format(sid=sensor_id, w=width, h=height, fps=fps),
            cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise IOError(f"CSI kamera acilamadi (sensor-id={sensor_id})")
        self.cam_id = sensor_id if cam_id is None else cam_id
        self.index = -1

    def read(self):
        ok, img = self.cap.read()
        if not ok: return None
        self.index += 1
        return Frame(img, time.monotonic(), self.index, self.cam_id)

    def release(self): self.cap.release()


class OnlineStabilizer:
    """
    frame -> downscale + gray -> LK tracking -> partial affine (RANSAC)
          -> cumulative trajectory -> EMA -> correction -> limit -> one warp
    """
    def __init__(self, cfg: StabConfig = StabConfig(), cam_id: int = 0):
        self.cfg, self.cam_id = cfg, cam_id
        if cfg.num_threads > 0: cv2.setNumThreads(cfg.num_threads)
        cv2.setUseOptimized(True)
        self._lk = dict(winSize=(cfg.lk_win, cfg.lk_win), maxLevel=cfg.lk_levels,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.traj = np.zeros(3)
        self.smooth = np.zeros(3)
        self.frame_count = 0
        self.scale = 1.0
        self.size = None
        self._center = (0.0, 0.0)
        self._zoom = 1.0 / (1.0 - 2.0 * self.cfg.crop_ratio)

    def _prepare(self, image):
        h, w = image.shape[:2]
        if self.size is None:
            self.size = (w, h)
            self.scale = min(1.0, self.cfg.proc_width / float(w))
            self._center = (w / 2.0, h / 2.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.scale < 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        return gray

    def _detect(self, gray, keep=None):
        """Detect features with a per-cell quota.

        goodFeaturesToTrack ranks corners globally, so one high-texture
        region -- a passing car, a close-up object -- can take most of the
        budget and drag RANSAC onto *its* motion instead of the camera's.
        Passing `keep` preserves surviving tracks and only fills the gaps."""
        cfg = self.cfg
        h, w = gray.shape
        gy = gx = cfg.grid
        per = max(4, cfg.max_corners // (gx*gy))
        mask = None
        if keep is not None and len(keep):
            mask = np.full((h, w), 255, np.uint8)
            for x, y in keep.reshape(-1, 2).astype(int):
                cv2.circle(mask, (x, y), cfg.min_distance, 0, -1)
        pts = []
        for i in range(gy):
            for j in range(gx):
                y0, y1 = i*h//gy, (i+1)*h//gy
                x0, x1 = j*w//gx, (j+1)*w//gx
                sub = None if mask is None else mask[y0:y1, x0:x1]
                p = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], per,
                        cfg.quality_level, cfg.min_distance,
                        mask=sub, blockSize=cfg.block_size)
                if p is not None:
                    p[:,0,0] += x0; p[:,0,1] += y0
                    pts.append(p)
        new = np.vstack(pts) if pts else None
        if keep is None or not len(keep):
            return new
        return keep if new is None else np.vstack([keep, new]).astype(np.float32)

    def _track(self, pg, cg, p0):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, p0, None, **self._lk)
        if p1 is None: return None, None
        st = st.reshape(-1).astype(bool)
        if self.cfg.use_fb_check and st.any():
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(cg, pg, p1, None, **self._lk)
            if pb is not None:
                err = np.linalg.norm(p0.reshape(-1, 2) - pb.reshape(-1, 2), axis=1)
                st &= st2.reshape(-1).astype(bool) & (err < self.cfg.fb_threshold)
        if st.sum() < 6: return None, None
        return p0[st], p1[st]

    def _estimate(self, p0, p1) -> Motion:
        M, inl = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_thresh,
            maxIters=500, confidence=0.99, refineIters=10)
        if M is None: return Motion()
        n = int(inl.sum()) if inl is not None else 0
        if n < self.cfg.min_inliers: return Motion(n_inliers=n)
        s = 1.0 / self.scale
        return Motion(float(M[0, 2]) * s, float(M[1, 2]) * s,
                      float(math.atan2(M[1, 0], M[0, 0])), True, n)

    def _fits(self, M, W, H):
        """Do all four output corners land inside the source frame?

        Checks rotation, zoom and translation together. Limiting each axis
        separately missed the extra margin rotation eats at the corners --
        at crop=0.10 and 6 degrees the frame overflowed by ~53 px and
        BORDER_REPLICATE smeared the edges."""
        Mi = cv2.invertAffineTransform(M)
        dst = np.array([[0,0],[W,0],[W,H],[0,H]], np.float32).reshape(-1,1,2)
        q = cv2.transform(dst, Mi).reshape(-1,2)
        return (q[:,0].min() >= 0 and q[:,1].min() >= 0
                and q[:,0].max() <= W and q[:,1].max() <= H)

    def _soft_limit(self, corr, mx, my, ma, knee=0.6):
        """Squash toward the limit with tanh past a knee instead of clipping.

        np.clip has a discontinuous derivative, so every time the correction
        hit the wall its velocity dropped to zero and the picture snapped --
        the most visible artifact this stabilizer produced."""
        lim = (mx, my, ma)
        for i in range(3):
            r = abs(corr[i]) / max(lim[i], 1e-9)
            if r > knee:
                t = (r - knee) / (1 - knee)
                corr[i] *= (knee + (1 - knee) * math.tanh(t)) / r
        return corr

    def _smooth_step(self, mx, my):
        cfg = self.cfg
        a = cfg.smooth_alpha
        self.smooth = a * self.smooth + (1 - a) * self.traj
        corr = self.smooth - self.traj
        if cfg.adaptive:
            r = max(abs(corr[0]) / max(mx, 1e-6), abs(corr[1]) / max(my, 1e-6))
            if r > 0.7:   # track the camera faster as we approach the limit
                ae = a - (a - 0.60) * min((r - 0.7) / 0.3, 1.0)
                self.smooth = ae * self.smooth + (1 - ae) * self.traj
                corr = self.smooth - self.traj
        ma = math.radians(cfg.max_angle_deg)
        corr = self._soft_limit(corr, mx, my, ma)
        # anti-windup: leak toward the limited value instead of snapping to it
        self.smooth += cfg.windup_leak * ((self.traj + corr) - self.smooth)
        return corr

    def _warp_matrix(self, corr):
        # Sign: estimateAffinePartial2D and getRotationMatrix2D disagree on
        # rotation direction, hence the minus. Without it the correction
        # roughly doubled the rotation instead of removing it.
        M = cv2.getRotationMatrix2D(self._center, -math.degrees(corr[2]), self._zoom)
        # The crop-zoom magnifies the image by z, so it magnifies the motion
        # inside it too. Without scaling the correction by z, a fraction
        # (z-1)/z of the shake survives -- 32% at crop=0.12.
        M[0, 2] += corr[0] * self._zoom
        M[1, 2] += corr[1] * self._zoom
        return M

    def process(self, frame: Frame, external_correction=None) -> StabilizedFrame:
        t0 = time.perf_counter()
        cfg = self.cfg
        gray = self._prepare(frame.image)
        W, H = self.size
        mx = cfg.crop_ratio * W * cfg.safety
        my = cfg.crop_ratio * H * cfg.safety

        motion = Motion()
        need = (self.prev_pts is None or len(self.prev_pts) < cfg.min_tracks
                or self.frame_count % cfg.redetect_interval == 0)
        if self.prev_gray is not None:
            if need: self.prev_pts = self._detect(self.prev_gray, keep=self.prev_pts)
            if self.prev_pts is not None and len(self.prev_pts) >= 6:
                p0, p1 = self._track(self.prev_gray, gray, self.prev_pts)
                if p0 is not None:
                    motion = self._estimate(p0, p1)
                    self.prev_pts = p1.reshape(-1, 1, 2).astype(np.float32)
                else:
                    self.prev_pts = None
            else:
                self.prev_pts = None

        if motion.valid:
    # Weight by inlier count rather than treating motion as all-or-nothing.
            # Assuming "the camera stopped" on a blurry frame made the
            # trajectory jump once tracking recovered.
            wgt = min(1.0, motion.n_inliers / float(cfg.conf_inliers))
            self.traj += wgt * motion.as_array()
        corr = self._smooth_step(mx, my)
        if external_correction is not None:
            corr = np.asarray(external_correction, dtype=np.float64)
            self.smooth = self.traj + corr

        M = self._warp_matrix(corr)
        for _ in range(6):                      # shrink until it fits
            if self._fits(M, W, H): break
            corr *= 0.85
            M = self._warp_matrix(corr)
        out = cv2.warpAffine(frame.image, M, (W, H),
                             flags=cfg.interpolation, borderMode=cfg.border_mode)
        self.prev_gray = gray
        self.frame_count += 1
        return StabilizedFrame(out, frame.timestamp, frame.index, frame.cam_id,
                               corr, motion, M,
                               0 if self.prev_pts is None else len(self.prev_pts),
                               (time.perf_counter() - t0) * 1000.0)


class CameraPipeline:
    """One source + one stabilizer. The sync layer will hold two of these."""
    def __init__(self, source, cfg=StabConfig(), cam_id=0):
        self.source, self.cam_id = source, cam_id
        self.stab = OnlineStabilizer(cfg, cam_id=cam_id)
    def next(self, external_correction=None):
        f = self.source.read()
        if f is None: return None
        f.cam_id = self.cam_id
        return self.stab.process(f, external_correction=external_correction)
    def release(self): self.source.release()


print("Core loaded.")
