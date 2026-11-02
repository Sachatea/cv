"""Метрики якості стабілізації.

З еталоном (синтетична тряска відомої траєкторії):
  * path_error  — середньоквадратична помилка залишкового руху відносно задуманої траєкторії P_t;
  * jitter_rms  — СКВ другої різниці траєкторії результату (прискорення, px/кадр²);
  * psnr_gt     — PSNR результату щодо «ідеального» кадру (без тремтіння).
Без еталона (реальні відео, незалежний від стабілізатора оцінювач Farnebäck):
  * itf         — Interframe Transformation Fidelity: середній PSNR між сусідніми кадрами;
  * stability   — частка енергії траєкторії результату в низьких частотах (Liu et al., 2013);
  * border      — частка кадрів із чорними краями (після корекції).
"""
import cv2
import numpy as np


def psnr(a, b, border=0.1):
    h, w = a.shape[:2]
    y0, x0 = int(h * border), int(w * border)
    a = a[y0:h - y0, x0:w - x0].astype(np.float32)
    b = b[y0:h - y0, x0:w - x0].astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def itf(frames, border=0.1):
    return float(np.mean([psnr(frames[i], frames[i + 1], border) for i in range(len(frames) - 1)]))


def global_path(frames, width=160):
    """Траєкторія результату (x, y, θ) незалежним щільним оцінювачем (Farnebäck)."""
    def prep(f):
        h, w = f.shape[:2]
        return cv2.cvtColor(cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
    prev = prep(frames[0])
    h, w = prev.shape
    gy, gx = np.mgrid[h // 8:h - h // 8:8, w // 8:w - w // 8:8]
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    c = np.array([w / 2, h / 2], np.float32)
    path = [np.zeros(3)]
    acc = np.zeros(3)
    for f in frames[1:]:
        cur = prep(f)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        v = flow[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        A, _ = cv2.estimateAffinePartial2D(pts - c, pts + v - c, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if A is not None:
            acc = acc + np.array([A[0, 2], A[1, 2], np.arctan2(A[1, 0], A[0, 0])])
        path.append(acc.copy())
        prev = cur
    return np.array(path)


def stability_score(frames):
    """Енергія 2–6-ї найнижчих частот траєкторії / сумарна (без DC), мінімум за x, y, θ."""
    p = global_path(frames)
    vals = []
    for k in range(3):
        sp = np.abs(np.fft.rfft(p[:, k] - p[:, k].mean())) ** 2
        sp = sp[1:]
        vals.append(sp[1:7].sum() / max(sp.sum(), 1e-12))
    return float(np.min(vals))


def border_exposure(frames, thr=3):
    """Частка кадрів, у яких на краях є «чорні» (майже нульові) смуги."""
    bad = 0
    for f in frames:
        g = f if f.ndim == 2 else f.mean(axis=2)
        edge = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
        bad += int(np.mean(edge < thr) > 0.2)
    return bad / len(frames)


def path_errors(gt, corr_full, rh_full):
    """gt: dict з P, J (T×4); corr_full: T×4 (dx, dy px; dθ рад; dlogs) застосована корекція.
    Повертає (path_error_px, rot_error_deg, jitter_rms_px)."""
    P, J = gt["P"], gt["J"]
    n = min(len(P), len(corr_full))
    R = (P + J)[:n] + corr_full[:n]          # залишкова траєкторія результату
    e = R - P[:n]
    err_xy = float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2)))
    err_rot = float(np.degrees(np.sqrt(np.mean(e[:, 2] ** 2))))
    acc = np.diff(R[:, :2], n=2, axis=0)
    jit = float(np.sqrt(np.mean(np.sum(acc ** 2, axis=1))))
    return err_xy, err_rot, jit


def path_jitter(frames, width=160):
    """RMS другої різниці (x, y) траєкторії, виміряної незалежним оцінювачем, у пікселях повного кадру.
    Працює без еталона: порівнюємо вхід і вихід. Менше = плавніше."""
    p = global_path(frames, width)
    k = frames[0].shape[1] / width
    a = np.diff(p[:, :2] * k, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(a ** 2, 1))))
