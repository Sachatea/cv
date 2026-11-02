"""Згладжування траєкторії камери.

Усі канали подано в «піксельних еквівалентах» (поворот і масштаб помножено на півдіагональ кадру),
тому параметри фільтрів однакові для всіх чотирьох каналів (x, y, поворот, масштаб).

Режими:
  * 'ma'      — каузальне ковзне середнє (бейзлайн);
  * 'kalman'  — фільтр Калмана зі сталою потужністю шуму процесу (бейзлайн);
  * 'intent'  — запропонований: фільтр Калмана, у якого потужність шуму процесу q адаптується
                до «намірного» руху камери (див. IntentKalman1D).
"""
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter1d


class IntentKalman1D:
    """Фільтр Калмана зі станом (позиція, швидкість) та моделлю сталої швидкості.

    Адаптація (intent-aware): за інновацією ν_t = z_t − H x_{t|t−1} оцінюються ковзне середнє
    m_t та ковзна потужність v_t. Відношення ρ = |m_t| / sqrt(v_t) ∈ [0, 1] близьке до 0 для
    тремтіння (інновації знакозмінні, нульове середнє) та до 1 для навмисного руху (панорамування:
    інновації мають стійкий знак). Потужність шуму процесу інтерполюється в логарифмічній шкалі
    між q_min (сильне згладжування) і q_max (швидке слідування):
        q = q_min · (q_max / q_min)^φ,  φ = clip((ρ' − ρ0)/(ρ1 − ρ0), 0, 1),
        ρ' = ρ · min(1, sqrt(v_t)/floor)   (щоб у нерухомому стані шум не вмикав слідування).
    """

    def __init__(self, r=1.0, q_min=0.002, q_max=0.5, alpha=0.15, rho0=0.35, rho1=0.8,
                 floor=0.6, adapt=True, vel_decay=0.9):
        self.r, self.q_min, self.q_max = r, q_min, q_max
        self.alpha, self.rho0, self.rho1, self.floor = alpha, rho0, rho1, floor
        self.adapt, self.vel_decay = adapt, vel_decay
        self.x = None

    def reset(self, z):
        self.x = np.array([z, 0.0])
        self.P = np.array([[self.r, 0.0], [0.0, 1.0]])
        self.m = 0.0
        self.v = 0.0
        self.q = self.q_min

    def _q(self):
        if not self.adapt:
            return self.q_min
        rho = abs(self.m) / np.sqrt(self.v + 1e-9)
        rho *= min(1.0, np.sqrt(self.v) / self.floor)
        phi = float(np.clip((rho - self.rho0) / (self.rho1 - self.rho0), 0.0, 1.0))
        return self.q_min * (self.q_max / self.q_min) ** phi

    def step(self, z, valid=True):
        if self.x is None:
            self.reset(z)
            return z
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        x = F @ self.x
        self.q = self._q()
        Q = self.q * np.array([[0.25, 0.5], [0.5, 1.0]])
        P = F @ self.P @ F.T + Q
        if valid:
            nu = z - x[0]
            self.m = (1 - self.alpha) * self.m + self.alpha * nu
            self.v = (1 - self.alpha) * self.v + self.alpha * nu * nu
            S = P[0, 0] + self.r
            K = np.array([P[0, 0] / S, P[1, 0] / S])
            x = x + K * nu
            P = P - np.outer(K, P[0, :])
        else:                                   # вимірювання ненадійне: лише прогноз із затуханням швидкості
            x[1] *= self.vel_decay
        self.x, self.P = x, P
        return float(self.x[0])


class PathSmoother:
    """Каузальний згладжувач 4-канальної траєкторії (x, y, поворот, масштаб)."""

    def __init__(self, mode="intent", q_min=0.002, q_max=0.5, r=1.0, ma_window=15, adapt=True, **kw):
        self.mode = mode
        self.ma_window = ma_window
        if mode in ("kalman", "intent"):
            self.f = [IntentKalman1D(r=r, q_min=q_min, q_max=q_max,
                                     adapt=(adapt and mode == "intent"), **kw) for _ in range(4)]
        elif mode == "ma":
            self.buf = deque(maxlen=ma_window)
        elif mode != "none":
            raise ValueError(mode)

    def reset(self, z):
        if self.mode in ("kalman", "intent"):
            for i, f in enumerate(self.f):
                f.reset(z[i])
        elif self.mode == "ma":
            self.buf.clear()
            self.buf.append(np.array(z, float))

    def step(self, z, valid=True):
        z = np.asarray(z, float)
        if self.mode == "none":
            return z.copy()
        if self.mode == "ma":
            self.buf.append(z)
            return np.mean(self.buf, axis=0)
        return np.array([f.step(z[i], valid) for i, f in enumerate(self.f)])

    def q_values(self):
        return [f.q for f in self.f] if self.mode in ("kalman", "intent") else []


def gaussian_smooth(path, sigma=15):
    """Некаузальний еталон: гаусове згладжування всієї траєкторії (path: T×4)."""
    return gaussian_filter1d(np.asarray(path, float), sigma=sigma, axis=0, mode="nearest")
