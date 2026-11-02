"""Оцінка глобального руху камери між двома сусідніми кадрами.

Етапи: (1) пошук кутових ознак із просторовим балансуванням (сітка, не більше k ознак
на комірку, щоб великий об'єкт на передньому плані не домінував); (2) піраміда Лукаса–Канаде
з перевіркою «вперед–назад»; (3) надійна оцінка перетворення подібності (зсув, поворот,
масштаб) методом RANSAC; (4) оцінка надійності вимірювання (inlier-ratio, кількість inliers,
охоплення кадру) та детектор склейки сцен.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionResult:
    params: np.ndarray        # (dx, dy, dtheta, dlogs): пікселі масштабованого кадру, радіани, ln(s)
    reliable: bool
    n_tracked: int
    n_inliers: int
    inlier_ratio: float
    scene_cut: bool


class MotionEstimator:
    def __init__(self, proc_width=320, max_corners=300, quality=0.01, min_dist=8,
                 grid=(6, 8), per_cell=4, fb_thresh=1.0, ransac_thresh=1.5,
                 min_inliers=12, min_ratio=0.35, min_spread=0.12, win=21, levels=3,
                 cut_thresh=30.0):
        self.proc_width = proc_width
        self.max_corners, self.quality, self.min_dist = max_corners, quality, min_dist
        self.grid, self.per_cell = grid, per_cell
        self.fb_thresh, self.ransac_thresh = fb_thresh, ransac_thresh
        self.min_inliers, self.min_ratio, self.min_spread = min_inliers, min_ratio, min_spread
        self.lk = dict(winSize=(win, win), maxLevel=levels,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.cut_thresh = cut_thresh
        self.prev = None
        self.size = None  # (w, h) масштабованого кадру
        self.scale = 1.0  # full_width / proc_width

    def reset(self):
        self.prev = None

    def prepare(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        pw = min(self.proc_width, w)
        self.scale = w / pw
        ph = int(round(h / self.scale))
        small = cv2.resize(frame_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        self.size = (pw, ph)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _features(self, gray):
        pts = cv2.goodFeaturesToTrack(gray, self.max_corners, self.quality, self.min_dist)
        if pts is None:
            return None
        pts = pts.reshape(-1, 2)
        h, w = gray.shape
        rows, cols = self.grid
        cell = (np.minimum((pts[:, 1] / h * rows).astype(int), rows - 1) * cols
                + np.minimum((pts[:, 0] / w * cols).astype(int), cols - 1))
        counts = {}
        keep = []
        for i, c in enumerate(cell):          # точки вже відсортовані за якістю
            n = counts.get(c, 0)
            if n < self.per_cell:
                counts[c] = n + 1
                keep.append(i)
        return pts[keep].astype(np.float32).reshape(-1, 1, 2)

    def estimate(self, gray):
        zero = np.zeros(4)
        if self.prev is None:
            self.prev = gray
            return MotionResult(zero, True, 0, 0, 1.0, False)
        prev = self.prev
        self.prev = gray
        h, w = gray.shape
        c = np.array([w / 2.0, h / 2.0], np.float32)

        p0 = self._features(prev)
        if p0 is None or len(p0) < 6:
            return self._fail(prev, gray, 0)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **self.lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **self.lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh)
        n_good = int(good.sum())
        if n_good < 6:
            return self._fail(prev, gray, n_good)
        a = p0.reshape(-1, 2)[good] - c
        b = p1.reshape(-1, 2)[good] - c
        A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_thresh,
                                             maxIters=500, confidence=0.99, refineIters=10)
        if A is None or inl is None:
            return self._fail(prev, gray, n_good)
        inl = inl.ravel().astype(bool)
        n_inl = int(inl.sum())
        ratio = n_inl / max(n_good, 1)
        # охоплення: площа ограничувальної рамки inliers відносно кадру
        if n_inl >= 3:
            pi = a[inl] + c
            spread = ((pi[:, 0].max() - pi[:, 0].min()) * (pi[:, 1].max() - pi[:, 1].min())) / (w * h)
        else:
            spread = 0.0
        reliable = (n_inl >= self.min_inliers and ratio >= self.min_ratio
                    and spread >= self.min_spread)
        s = float(np.hypot(A[0, 0], A[1, 0]))
        th = float(np.arctan2(A[1, 0], A[0, 0]))
        params = np.array([A[0, 2], A[1, 2], th, np.log(max(s, 1e-6))])
        if not reliable:
            return self._fail(prev, gray, n_good, n_inl, ratio)
        return MotionResult(params, True, n_good, n_inl, ratio, False)

    def _fail(self, prev, gray, n_good, n_inl=0, ratio=0.0):
        small_a = cv2.resize(prev, (64, 48), interpolation=cv2.INTER_AREA)
        small_b = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)
        cut = float(np.mean(cv2.absdiff(small_a, small_b))) > self.cut_thresh
        return MotionResult(np.zeros(4), False, n_good, n_inl, ratio, cut)
