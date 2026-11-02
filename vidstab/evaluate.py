"""Запуск методів стабілізації на кліпі та підрахунок усіх метрик."""
from dataclasses import replace

import numpy as np
import cv2

from .filters import gaussian_smooth
from .metrics import (psnr, itf, stability_score, border_exposure, path_errors, path_jitter)
from .stabilizer import OnlineStabilizer, StabConfig


def run_online(frames, cfg):
    stab = OnlineStabilizer(cfg)
    out = [stab.process(f) for f in frames]
    corr_full = np.array([[d[0] * stab.f, d[1] * stab.f, d[2] / stab.rh, d[3] / stab.rh]
                          for d in stab.history["corr"]])
    ms = 1000 * (stab.timing["estimate"] + stab.timing["filter"] + stab.timing["warp"])
    return out, stab, corr_full, ms


def run_offline_gauss(frames, cfg, sigma=15):
    """Некаузальний еталон: гаусове згладжування всієї траєкторії руху (усі кадри відомі наперед)."""
    out0, stab, _, _ = run_online(frames, replace(cfg, method="none"))
    C = np.array(stab.history["C"])
    S = gaussian_smooth(C, sigma=sigma)
    d = S - C
    d = stab.margin * np.tanh(d / stab.margin)
    corr_full = np.array([[x[0] * stab.f, x[1] * stab.f, x[2] / stab.rh, x[3] / stab.rh] for x in d])
    # перераховуємо zoom для цього методу
    from .stabilizer import required_zoom
    pw, ph = stab.est.size
    th = stab.margin[2] / stab.rh
    stab.zoom = required_zoom(pw, ph, stab.margin[0], stab.margin[1], th, th)
    out = [cv2.warpAffine(f, stab._matrix(c), (stab.W, stab.H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE) for f, c in zip(frames, corr_full)]
    return out, stab, corr_full, 0.0


def evaluate(frames, ideal, gt, name, cfg, offline=False, jit_in=None):
    if offline:
        out, stab, corr_full, ms = run_offline_gauss(frames, cfg)
    else:
        out, stab, corr_full, ms = run_online(frames, cfg)
    rh_full = 0.5 * np.hypot(stab.W, stab.H)
    ex, er, jit = path_errors(gt, corr_full, rh_full) if gt is not None else (np.nan,) * 3
    ps = np.nan
    zt = np.array(stab.history["zoom"]) if len(stab.history["zoom"]) else np.full(len(frames), stab.zoom)
    if offline:
        zt = np.full(len(frames), stab.zoom)
    if ideal is not None:
        vals = []
        for o, i, z in zip(out, ideal, zt):
            stab.zoom = z
            vals.append(psnr(o, stab.zoom_only(i)))
        ps = float(np.mean(vals))
    zjit = float(np.sqrt(np.mean(np.diff(np.log(zt)) ** 2))) if len(zt) > 2 else 0.0
    return dict(method=name, path_err_px=ex, rot_err_deg=er, jitter_px=jit, psnr_gt=ps,
                itf=itf(out), jit_out=(path_jitter(out) if gt is None else np.nan),
                jit_in=(jit_in if jit_in is not None else (path_jitter(frames) if gt is None else np.nan)), border=border_exposure(out),
                crop_area=float(np.mean(1.0 / zt ** 2)), zoom_jitter=zjit, ms_per_frame=ms,
                fps=(1000.0 / ms if ms > 0 else np.nan),
                reliable_frac=float(np.mean(stab.history["reliable"])))
