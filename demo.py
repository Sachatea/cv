#!/usr/bin/env python3
"""Інтерактивне демо стабілізації відео в реальному часі (вебкамера або файл).

  python3 demo.py                      # вебкамера 0
  python3 demo.py --source video.mp4   # файл
  python3 demo.py --source video.mp4 --out stabilized.mp4 --no-gui   # без вікна (пакетно)

Клавіші: SPACE — увімк/вимк стабілізацію · M — змінити метод (intent/kalman/ma) · P — пауза ·
         R — скинути траєкторію · S — зберегти знімок · Q/Esc — вихід.
Повзунки: Smooth (сила згладжування), Crop % (запас на корекцію; змінює зум), Compare (0 = лише результат).
"""
import argparse, os, sys, time
import cv2, numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidstab.stabilizer import OnlineStabilizer, StabConfig

METHODS = ["kalman", "intent", "ma"]


def q_from_slider(v):          # 0..10 -> q_min від 1e-4 (слабко) до 1e-7 (сильно)
    return 10 ** (-4 - 0.3 * v)


def draw_hud(img, text_lines, hist, w):
    for i, t in enumerate(text_lines):
        cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    # міні-графік x-траєкторії: сира (червона) та згладжена (зелена)
    if len(hist["C"]) > 2:
        gh, gw = 60, 200
        x0, y0 = w - gw - 10, 10
        cv2.rectangle(img, (x0, y0), (x0 + gw, y0 + gh), (30, 30, 30), -1)
        C = np.array(hist["C"][-gw:])[:, 0]; S = np.array(hist["S"][-gw:])[:, 0]
        lo, hi = min(C.min(), S.min()) - 1, max(C.max(), S.max()) + 1
        for arr, col in ((C, (60, 60, 255)), (S, (60, 255, 60))):
            pts = np.stack([x0 + np.arange(len(arr)), y0 + gh - (arr - lo) / (hi - lo) * gh], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=640, help="ширина обробки/виводу")
    ap.add_argument("--out"); ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--crop", type=float, default=0.04)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Не вдалося відкрити джерело: %s (macOS: дозвольте доступ до камери для Terminal)" % a.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    cfg = StabConfig(crop=a.crop)
    stab, writer = None, None
    enabled, paused, mi, adaptive = True, False, 0, True
    sm_val, crop_val = 6, int(round(a.crop * 100))
    win = "Video stabilization"
    if not a.no_gui:
        cv2.namedWindow(win)
        cv2.createTrackbar("Smooth", win, sm_val, 10, lambda v: None)
        cv2.createTrackbar("Crop %", win, crop_val, 10, lambda v: None)
        cv2.createTrackbar("Compare", win, 1, 1, lambda v: None)
    fps_ema, t_prev, n = 0.0, time.time(), 0
    frame = None
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w != a.width:
                frame = cv2.resize(frame, (a.width, int(round(h * a.width / w / 2)) * 2), interpolation=cv2.INTER_AREA)
        if not a.no_gui:
            nsm, ncr = cv2.getTrackbarPos("Smooth", win), max(1, cv2.getTrackbarPos("Crop %", win))
            if stab is None or nsm != sm_val or ncr != crop_val:
                sm_val, crop_val = nsm, ncr
                stab = None
        if stab is None:
            stab = OnlineStabilizer(replace(cfg, method=METHODS[mi], crop=crop_val / 100, adaptive_crop=adaptive, q_min=q_from_slider(sm_val),
                                           q_max=q_from_slider(sm_val) * 1000))
        t0 = time.perf_counter()
        out = stab.process(frame) if enabled else frame
        dt = time.perf_counter() - t0
        n += 1
        now = time.time(); fps_ema = 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-6) if n > 1 else 0; t_prev = now
        view = out.copy()
        lines = ["%s | %s | proc %.1f ms (%.0f fps)  e2e %.0f fps" % (
            "STAB ON" if enabled else "STAB OFF", METHODS[mi], dt * 1000, 1 / max(dt, 1e-6), fps_ema),
            "zoom %.3f  area %.0f%%  crop %s" % (stab.zoom, 100 / stab.zoom ** 2, "adaptive" if stab.cfg.adaptive_crop else "fixed")]
        if enabled and stab.history["reliable"]:
            lines.append("tracking: %s" % ("OK" if stab.history["reliable"][-1] else "LOST (hold)"))
        draw_hud(view, lines, stab.history, view.shape[1])
        show = np.hstack([frame, view]) if (a.no_gui or cv2.getTrackbarPos("Compare", win)) else view
        if a.out:
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (show.shape[1], show.shape[0]))
            writer.write(show)
        if not a.no_gui:
            cv2.imshow(win, show)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")): break
            elif k == ord(" "): enabled = not enabled
            elif k == ord("m"): mi = (mi + 1) % len(METHODS); stab = None
            elif k == ord("p"): paused = not paused
            elif k == ord("r"): stab = None
            elif k == ord("a"): adaptive = not adaptive; stab = None
            elif k == ord("s"): cv2.imwrite("snapshot_%d.png" % int(time.time()), show)
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()
    if writer: writer.release()
    if not a.no_gui: cv2.destroyAllWindows()
    print("кадрів: %d, середній час обробки (оцінка/фільтр/warp, EMA мс): %s" % (
        n, {k: round(v * 1000, 2) for k, v in stab.timing.items()}))


if __name__ == "__main__":
    main()
