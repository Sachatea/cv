"""Синтетичні відео з ВІДОМОЮ траєкторією камери (еталоном) для кількісної оцінки.

Камера = P_t (задуманий рух: нерухома / панорама) + J_t (тремтіння). Для кожного кадру
відомі і «тремтяче» зображення, і «ідеальне» (лише P_t), тому можна рахувати помилку
траєкторії та PSNR щодо еталона. Те саме працює й для користувацького відео (make_dataset.py).
"""
import numpy as np
import cv2

CATEGORIES = {
    # звичайні режими
    "low":        dict(shake="low"),
    "medium":     dict(shake="medium"),
    "high":       dict(shake="high"),
    "walking":    dict(shake="walking", pan="slow"),
    "impulse":    dict(shake="bumps"),
    # найгірші входи (worst case)
    "lowtex":     dict(shake="medium", lowtex=True),
    "occlusion":  dict(shake="medium", fg=0.45),
    "blur_noise": dict(shake="high", blur=True, noise=7.0),
    "fastpan":    dict(shake="medium", pan="fast"),
    "scenecut":   dict(shake="medium", cut=True),
}
SHAKE = {   # (σ_xy px, σ_rot градуси) на кадрі 640×480
    "low": (3.0, 0.2), "medium": (8.0, 0.6), "high": (18.0, 1.5),
}
TREMOR_SPEED, SWAY_SPEED, SWAY_FRAC = 0.30, 0.04, 0.30   # швидкий тремор (~1.4 Гц при 30 к/с) + повільне хитання


EXTRA = {   # додаткові сценарії (не входять до основного набору)
    "mixed": dict(shake="mixed"),   # нестаціонарне тремтіння: low -> high -> low (по третині кліпу)
}


def make_canvas(seed, size=(1920, 1440), lowtex=False):
    rng = np.random.default_rng(seed)
    W, H = size
    if lowtex:   # майже однорідний фон: плавний градієнт + кілька слабких плям
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        base = 120 + 25 * np.sin(xx / 900 + seed) + 20 * np.cos(yy / 700)
        img = np.stack([base, base * 0.95, base * 1.05], -1)
        for _ in range(6):
            cv2.circle(img, (int(rng.integers(W)), int(rng.integers(H))), int(rng.integers(30, 80)),
                       [float(x) for x in rng.integers(100, 150, 3)], -1)
        img = cv2.GaussianBlur(img, (0, 0), 6)
        return np.clip(img, 0, 255).astype(np.uint8)
    img = np.zeros((H, W, 3), np.float32)
    for octave, w in enumerate((8, 16, 32, 64, 128, 256)):   # багатомасштабний шум
        n = rng.normal(0, 1, (int(w * H / W) + 1, w, 3)).astype(np.float32)
        img += cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) * (60 / (1 + 0.45 * octave))
    img = img - img.min()
    img = (img / img.max() * 200 + 25).astype(np.uint8)
    for _ in range(140):         # геометричні фігури та текст
        col = tuple(int(v) for v in rng.integers(20, 235, 3))
        x, y = int(rng.integers(W)), int(rng.integers(H))
        k = rng.integers(3)
        if k == 0:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 120)), y + int(rng.integers(20, 120))), col, -1)
        elif k == 1:
            cv2.circle(img, (x, y), int(rng.integers(10, 60)), col, 2)
        else:
            cv2.putText(img, "VS%d" % rng.integers(100, 999), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        float(rng.uniform(0.6, 1.6)), col, 2)
    return img


def ou(T, sigma, speed, rng):
    """Процес Орнштейна–Уленбека зі стаціонарним σ."""
    x = np.zeros(T)
    s = sigma * np.sqrt(2 * speed - speed * speed)
    for t in range(1, T):
        x[t] = x[t - 1] * (1 - speed) + s * rng.normal()
    return x


def shake_path(kind, T, rng):
    """J_t: T×4 (dx px, dy px, θ рад, ln s)."""
    J = np.zeros((T, 4))
    if kind == "mixed":
        n = T // 3
        segs = [shake_path(k, m, rng) for k, m in (("low", n), ("high", n), ("low", T - 2 * n))]
        off = np.zeros(4)
        parts = []
        for sg in segs:      # безперервність на стиках
            sg = sg + (off - sg[0])
            parts.append(sg); off = sg[-1]
        return np.vstack(parts)
    if kind in SHAKE:
        sxy, srot = SHAKE[kind]
        for ch, sg in ((0, sxy), (1, sxy), (2, np.deg2rad(srot)), (3, 0.002)):
            J[:, ch] = ou(T, sg, TREMOR_SPEED, rng) + ou(T, sg * SWAY_FRAC, SWAY_SPEED, rng)
    elif kind == "walking":      # кроки ≈ 2 Гц при 30 к/с: вертикальне підстрибування
        t = np.arange(T)
        ph = rng.uniform(0, 6.28)
        J[:, 1] = 10 * np.sin(2 * np.pi * t / 15 + ph) + ou(T, 2.0, 0.1, rng)
        J[:, 0] = 4 * np.sin(2 * np.pi * t / 30 + ph) + ou(T, 2.0, 0.1, rng)
        J[:, 2] = np.deg2rad(0.6 * np.sin(2 * np.pi * t / 30 + ph)) + np.deg2rad(ou(T, 0.2, 0.1, rng))
    elif kind == "bumps":        # імпульсні поштовхи, що затухають
        J[:, 0] = ou(T, 2.0, 0.1, rng); J[:, 1] = ou(T, 2.0, 0.1, rng)
        t = 20
        while t < T:
            amp = rng.normal(0, 16, 2)
            L = min(12, T - t)
            for k in range(L):
                J[t + k, :2] += amp * np.exp(-k / 3.5)
            t += int(rng.integers(30, 60))
    return J


def intent_path(kind, T):
    """P_t: T×4, задуманий рух камери."""
    P = np.zeros((T, 4))
    if kind in ("slow", "fast"):
        v = 1.2 if kind == "slow" else 4.0
        t = np.arange(T)
        ramp = np.minimum(1.0, np.minimum(t, T - 1 - t) / 20.0)
        P[:, 0] = np.cumsum(v * ramp)
        P[:, 1] = 0.25 * np.cumsum(v * ramp)
    return P


def render_clip(category, seed=0, T=150, size=(640, 480)):
    """Повертає (shaken, ideal, gt) для категорії з CATEGORIES."""
    spec = CATEGORIES.get(category) or EXTRA[category]
    rng = np.random.default_rng(seed)
    W, H = size
    canvas = make_canvas(seed, lowtex=spec.get("lowtex", False))
    canvas2 = make_canvas(seed + 999) if spec.get("cut") else None
    P = intent_path(spec.get("pan"), T)
    J = shake_path(spec["shake"], T, rng)
    fg_frac = spec.get("fg", 0.0)
    cw, ch = canvas.shape[1], canvas.shape[0]
    cc = np.array([cw / 2, ch / 2])
    objs = [dict(pos=rng.uniform([cw * .35, ch * .35], [cw * .65, ch * .65]), vel=rng.normal(0, 2.5, 2),
                 size=int(0.5 * np.sqrt(fg_frac * W * H) if fg_frac else 50),
                 col=tuple(int(v) for v in rng.integers(30, 230, 3))) for _ in range(2 if fg_frac == 0 else 1)]

    def view(c, path, t, extra=None):
        dx, dy, th, ls = path
        s = np.exp(ls)
        RS = np.array([[s * np.cos(th), -s * np.sin(th)], [s * np.sin(th), s * np.cos(th)]])
        tvec = np.array([W / 2, H / 2]) + np.array([dx, dy]) - RS @ cc
        M = np.hstack([RS, tvec[:, None]])
        return cv2.warpAffine(c, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    shaken, ideal = [], []
    for t in range(T):
        base = canvas2 if (canvas2 is not None and t >= T // 2) else canvas
        c = base.copy()
        for o in objs:
            o["pos"] = o["pos"] + o["vel"] + P[t, :2] * 0.0
            p = o["pos"].astype(int); sz = o["size"]
            cv2.rectangle(c, (p[0] - sz, p[1] - sz), (p[0] + sz, p[1] + sz), o["col"], -1)
            cv2.circle(c, (p[0], p[1]), sz // 2, (240, 240, 240), 3)
        f_s = view(c, P[t] + J[t], t)
        f_i = view(c, P[t], t)
        if spec.get("blur") and t > 0:
            sp = np.linalg.norm(J[t, :2] - J[t - 1, :2])
            L = int(np.clip(sp * 0.7, 1, 25))
            if L > 2:
                ang = np.arctan2(J[t, 1] - J[t - 1, 1], J[t, 0] - J[t - 1, 0])
                k = np.zeros((L, L), np.float32)
                cv2.line(k, (0, L // 2), (L - 1, L // 2), 1.0, 1)
                Mr = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), np.degrees(ang), 1)
                k = cv2.warpAffine(k, Mr, (L, L))
                f_s = cv2.filter2D(f_s, -1, k / max(k.sum(), 1e-6))
        if spec.get("noise"):
            f_s = np.clip(f_s.astype(np.float32) + rng.normal(0, spec["noise"], f_s.shape), 0, 255).astype(np.uint8)
        shaken.append(f_s); ideal.append(f_i)
    return shaken, ideal, dict(P=P, J=J, category=category, seed=seed, size=size)


def shake_real_video(frames, kind="medium", seed=0):
    """Додає відомий тремор J_t до СТАБІЛЬНОГО відео користувача (штатив / нерухома камера).
    Еталон: ideal = вихідні кадри, P_t = 0. Повертає (shaken, ideal, gt). Краї — відображення."""
    T = len(frames)
    H, W = frames[0].shape[:2]
    rng = np.random.default_rng(seed)
    k = W / 640.0                         # амплітуди визначені для кадру 640 px
    J = shake_path(kind, T, rng)
    J[:, :2] *= k
    c = np.array([W / 2, H / 2])
    out = []
    for f, (dx, dy, th, ls) in zip(frames, J):
        s = np.exp(ls)
        RS = np.array([[s * np.cos(th), -s * np.sin(th)], [s * np.sin(th), s * np.cos(th)]])
        M = np.hstack([RS, (c + [dx, dy] - RS @ c)[:, None]])
        out.append(cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return out, list(frames), dict(P=np.zeros((T, 4)), J=J, category="real_" + kind, seed=seed, size=(W, H))
