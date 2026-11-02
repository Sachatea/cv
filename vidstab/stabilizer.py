"""Онлайн-стабілізатор: кадр на вході -> кадр на виході, без затримки (каузальний)."""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .filters import PathSmoother
from .motion import MotionEstimator


@dataclass
class StabConfig:
    method: str = "kalman"        # 'intent' | 'kalman' | 'ma' | 'none'
    crop: float = 0.04            # допустимий зсув корекції як частка сторони кадру
    proc_width: int = 320         # ширина кадру для оцінки руху
    q_min: float = 1e-6
    q_max: float = 1e-3
    r: float = 1.0
    ma_window: int = 15
    soft_margin: bool = False     # tanh-обмеження корекції (інакше жорстке np.clip)
    gating: bool = False          # відкидання ненадійних вимірювань + скидання при склейці
    adapt: bool = True            # адаптація q до намірного руху
    alpha: float = 0.1
    rho0: float = 0.4
    rho1: float = 0.8
    floor: float = 1.0
    adaptive_crop: bool = True    # змінний (повільно) запас корекції і зум за амплітудою тремтіння
    crop_min: float = 0.03
    crop_max: float = 0.08
    crop_gain: float = 1.0        # запас відносно піка потрібної корекції
    crop_decay: float = 0.995     # спад піка потрібної корекції за кадр (утримання піка)
    crop_rate_up: float = 0.002   # макс. приріст κ за кадр
    crop_rate_down: float = 0.0004  # макс. спад κ за кадр
    soft_weight: bool = True      # шум вимірювання R_t залежить від якості оцінки руху
    weight_min: float = 0.1
    ecc_fallback: bool = True     # запасний ECC-оцінювач на слабкій текстурі
    unreliable_reset_after: int = 24   # кадрів поспіль без надійного руху -> м'яке скидання


def required_zoom(W, H, mx, my, th_max, ls_max):
    """Мінімальне масштабування, за якого корекція з |dx|≤mx, |dy|≤my, |θ|≤th_max,
    |ln s|≤ls_max гарантовано не відкриває чорних країв."""
    corners = np.array([[sx * W / 2, sy * H / 2] for sx in (-1, 1) for sy in (-1, 1)])
    worst = 1.0
    for dx in (-mx, mx):
        for dy in (-my, my):
            for th in (-th_max, th_max):
                for ls in (-ls_max, ls_max):
                    s = np.exp(ls)
                    c, sn = np.cos(-th), np.sin(-th)
                    R = np.array([[c, -sn], [sn, c]])
                    lo, hi = 1.0, 3.0
                    for _ in range(40):
                        z = 0.5 * (lo + hi)
                        q = (corners / z - np.array([dx, dy])) @ R.T / s
                        ok = np.all(np.abs(q[:, 0]) <= W / 2 + 1e-6) and np.all(np.abs(q[:, 1]) <= H / 2 + 1e-6)
                        if ok:
                            hi = z
                        else:
                            lo = z
                    worst = max(worst, hi)
    return worst


class OnlineStabilizer:
    def __init__(self, cfg=None):
        self.cfg = cfg or StabConfig()
        self.est = MotionEstimator(proc_width=self.cfg.proc_width, ecc_fallback=self.cfg.ecc_fallback)
        self.smoother = None
        self.t = 0
        self.timing = dict(estimate=0.0, filter=0.0, warp=0.0)
        self.history = dict(C=[], S=[], corr=[], reliable=[], cut=[], q=[], zoom=[])
        self._bad = 0

    # --- ініціалізація за розміром першого кадру
    def _init(self, frame):
        H, W = frame.shape[:2]
        self.W, self.H = W, H
        gray = self.est.prepare(frame)
        pw, ph = self.est.size
        self.f = self.est.scale
        self.rh = 0.5 * np.hypot(pw, ph)                  # півдіагональ масштабованого кадру
        c = self.cfg
        m_xy = np.array([c.crop * pw, c.crop * ph])
        m_r = 0.5 * c.crop * min(pw, ph)
        self.margin = np.array([m_xy[0], m_xy[1], m_r, m_r])   # у «піксельних еквівалентах»
        th_max = m_r / self.rh
        self.zoom = 1.0 if c.method == "none" else required_zoom(
            pw, ph, m_xy[0], m_xy[1], th_max, th_max)
        self.ref = np.array([pw, ph, 0.5 * min(pw, ph), 0.5 * min(pw, ph)])   # margin = κ · ref
        if c.adaptive_crop and c.method != "none":
            self.kappas = np.linspace(c.crop_min, c.crop_max, 25)
            self.zooms = np.array([required_zoom(pw, ph, k * pw, k * ph, 0.5 * k * min(pw, ph) / self.rh,
                                                 0.5 * k * min(pw, ph) / self.rh) for k in self.kappas])
            self.kappa = float(np.clip(c.crop, c.crop_min, c.crop_max))
            self.peak = self.kappa / c.crop_gain
        self.smoother = PathSmoother(c.method, q_min=c.q_min, q_max=c.q_max, r=c.r,
                                     ma_window=c.ma_window, adapt=c.adapt,
                                     alpha=c.alpha, rho0=c.rho0, rho1=c.rho1, floor=c.floor)
        self.C = np.zeros(4)
        self.smoother.reset(self.C)
        return gray

    def _adapt_crop(self, d_raw):
        """Онлайн-адаптація κ: піковий (із повільним спадом) потрібний запас → цільове κ → обмеження швидкості."""
        c = self.cfg
        need = float(np.max(np.abs(d_raw) / self.ref))
        self.peak = max(need, c.crop_decay * self.peak)
        target = float(np.clip(c.crop_gain * self.peak, c.crop_min, c.crop_max))
        self.kappa += float(np.clip(target - self.kappa, -c.crop_rate_down, c.crop_rate_up))
        self.kappa = float(np.clip(self.kappa, c.crop_min, c.crop_max))
        self.margin = self.kappa * self.ref
        self.zoom = float(np.interp(self.kappa, self.kappas, self.zooms))

    def _matrix(self, corr):
        """corr: (dx, dy, dθ, dlogs) для повного кадру -> афінна 2×3 з масштабуванням zoom."""
        dx, dy, th, ls = corr
        s = np.exp(ls)
        cx, cy = self.W / 2.0, self.H / 2.0
        T = lambda x, y: np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]])
        RS = np.array([[s * np.cos(th), -s * np.sin(th), 0], [s * np.sin(th), s * np.cos(th), 0], [0, 0, 1.0]])
        Z = np.array([[self.zoom, 0, 0], [0, self.zoom, 0], [0, 0, 1.0]])
        M = T(cx, cy) @ Z @ T(dx, dy) @ RS @ T(-cx, -cy)
        return M[:2]

    def process(self, frame):
        cfg = self.cfg
        t0 = time.perf_counter()
        if self.t == 0:
            gray = self._init(frame)
        else:
            gray = self.est.prepare(frame)
        res = self.est.estimate(gray)
        t1 = time.perf_counter()

        reliable = True if self.t == 0 else res.reliable
        params = res.params.copy()
        if cfg.gating:
            if res.scene_cut:
                self.smoother.reset(self.C)              # нова сцена: корекція скидається
                self._bad = 0
            if not reliable:
                params[:] = 0.0                          # руху не знаємо -> вважаємо камеру нерухомою
                self._bad += 1
                if self._bad >= cfg.unreliable_reset_after:
                    self.smoother.reset(self.C)
                    self._bad = 0
            else:
                self._bad = 0
        pe = np.array([params[0], params[1], params[2] * self.rh, params[3] * self.rh])
        self.C = self.C + pe
        valid = reliable if cfg.gating else True
        r_scale = 1.0
        if cfg.soft_weight and reliable and self.t > 0:
            w = float(np.clip(res.inlier_ratio, 0.3, 1.0) * min(1.0, res.n_inliers / 40.0)) if res.n_inliers else float(np.clip(res.inlier_ratio, 0.3, 1.0))
            r_scale = 1.0 / max(w, cfg.weight_min) ** 2
        S = self.smoother.step(self.C, valid, r_scale)
        d = S - self.C
        if cfg.method == "none":
            d = np.zeros(4)
        elif cfg.adaptive_crop:
            self._adapt_crop(d)
            d = np.clip(d, -self.margin, self.margin)
        elif cfg.soft_margin:
            d = self.margin * np.tanh(d / self.margin)
        else:
            d = np.clip(d, -self.margin, self.margin)
        t2 = time.perf_counter()

        corr_full = (d[0] * self.f, d[1] * self.f, d[2] / self.rh, d[3] / self.rh)
        if cfg.method == "none" and self.zoom == 1.0:
            out = frame
        else:
            out = cv2.warpAffine(frame, self._matrix(corr_full), (self.W, self.H),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        t3 = time.perf_counter()
        a = 0.1
        for k, v in zip(("estimate", "filter", "warp"), (t1 - t0, t2 - t1, t3 - t2)):
            self.timing[k] = v if self.t == 0 else (1 - a) * self.timing[k] + a * v
        h = self.history
        h["C"].append(self.C.copy()); h["S"].append(S.copy()); h["corr"].append(d.copy())
        h["reliable"].append(bool(reliable)); h["cut"].append(bool(res.scene_cut))
        h["zoom"].append(self.zoom)
        h["q"].append(self.smoother.q_values()[:1] or [0.0])
        self.t += 1
        return out

    def zoom_only(self, frame):
        """Той самий zoom без корекції (для порівняння з еталонним кадром)."""
        return cv2.warpAffine(frame, self._matrix((0, 0, 0, 0)), (self.W, self.H),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
