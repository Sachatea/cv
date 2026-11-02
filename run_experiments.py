#!/usr/bin/env python3
"""Експерименти: порівняння методів стабілізації на збалансованому наборі.

Дані:  (1) синтетичні кліпи з відомим еталоном — 10 категорій (5 звичайних + 5 worst-case) × N кліпів × T кадрів;
       (2) (опційно, --stable_videos DIR) власні СТАБІЛЬНІ відео + доданий відомий тремор (еталон відомий);
       (3) (опційно, --shaky_videos DIR) власні реальні «тремтячі» відео (без еталона: ITF та jitter-ratio).
Результат: results/per_clip.csv, summary_by_method.csv, summary_by_category.csv, wilcoxon.csv, figures/*.png
Приклад:  python3 run_experiments.py --seeds 200 201 202 203 --T 150
"""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd, cv2
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidstab.stabilizer import StabConfig
from vidstab.synth import CATEGORIES, render_clip, shake_real_video
from vidstab.evaluate import evaluate
from vidstab.metrics import path_jitter

# Гіперпараметри базових методів підібрано на ОКРЕМИХ seed 100, 101 (tools/tune_smoothers.py)
BASE = StabConfig()   # фінальний метод: Калман + адаптивний запас/зум + ECC-запас + м'яка вага вимірювань
OLD = replace(BASE, adaptive_crop=False, ecc_fallback=False, soft_weight=False)   # попередній «фінальний» (фіксований κ = 0.04)
METHODS = {
    "none":               replace(BASE, method="none"),
    "MA-9":               replace(OLD, method="ma", ma_window=9),
    "Kalman (fixed crop)": OLD,
    "Final (ours)":       BASE,
    # абляції фінального методу (по одній складовій)
    "-adaptive crop":     replace(BASE, adaptive_crop=False),
    "-ECC fallback":      replace(BASE, ecc_fallback=False),
    "-soft weight":       replace(BASE, soft_weight=False),
    # фіксований κ (з ECC і вагою) — для Парето-кривої «площа кадру — якість»
    "fixed k=0.02":       replace(BASE, adaptive_crop=False, crop=0.02),
    "fixed k=0.03":       replace(BASE, adaptive_crop=False, crop=0.03),
    "fixed k=0.04":       replace(BASE, adaptive_crop=False, crop=0.04),
    "fixed k=0.06":       replace(BASE, adaptive_crop=False, crop=0.06),
    "fixed k=0.08":       replace(BASE, adaptive_crop=False, crop=0.08),
}


def read_video(path, max_frames=300, width=640):
    cap = cv2.VideoCapture(path); fr = []
    while len(fr) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if w != width:
            f = cv2.resize(f, (width, int(round(h * width / w / 2)) * 2), interpolation=cv2.INTER_AREA)
        fr.append(f)
    return fr


def clips(args):
    for cat in CATEGORIES:
        for s in args.seeds:
            sh, ideal, gt = render_clip(cat, s, T=args.T)
            yield cat, f"{cat}_{s}", sh, ideal, gt
    for p in sorted(glob.glob(os.path.join(args.stable_videos, "*"))) if args.stable_videos else []:
        base = read_video(p, args.T)
        for kind in ("low", "medium", "high", "walking"):
            sh, ideal, gt = shake_real_video(base, kind, seed=1)
            yield "real_" + kind, f"{os.path.basename(p)}_{kind}", sh, ideal, gt
    for p in sorted(glob.glob(os.path.join(args.shaky_videos, "*"))) if args.shaky_videos else []:
        yield "real_shaky", os.path.basename(p), read_video(p, args.T), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[400, 401, 402, 403, 404, 405, 406, 407])
    ap.add_argument("--T", type=int, default=150)
    ap.add_argument("--stable_videos"); ap.add_argument("--shaky_videos")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    rows = []; frames_total = 0; t0 = time.time()
    for cat, cid, sh, ideal, gt in clips(args):
        frames_total += len(sh)
        jin = path_jitter(sh) if gt is None else None
        for name, cfg in METHODS.items():
            r = evaluate(sh, ideal, gt, name, cfg, jit_in=jin); r.update(category=cat, clip=cid, frames=len(sh)); rows.append(r)
        r = evaluate(sh, ideal, gt, "Gauss offline (ref)", BASE, offline=True)
        r.update(category=cat, clip=cid, frames=len(sh)); rows.append(r)
        print(f"{cid:28s} done  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "per_clip.csv"), index=False)
    print(f"\nУсього кадрів у наборі: {frames_total}, кліпів: {df["clip"].nunique()}")
    cols = ["path_err_px", "rot_err_deg", "jitter_px", "psnr_gt", "itf", "jit_out", "jit_in", "border", "crop_area", "zoom_jitter", "fps"]
    df.groupby("method")[cols].mean().round(3).to_csv(os.path.join(args.out, "summary_by_method.csv"))
    df.pivot_table(index="method", columns="category", values="path_err_px").round(2).to_csv(
        os.path.join(args.out, "summary_by_category.csv"))
    print(df.groupby("method")[cols].mean().round(2).to_string())
    # Парний тест Вілкоксона (за кліпами) для path_err_px та jitter_px: Intent vs інші
    try:
        from scipy.stats import wilcoxon
        out = []
        for m in df.method.unique():
            if m == "Final (ours)":
                continue
            for metric in ("path_err_px", "jitter_px"):
                a = df[df.method == "Final (ours)"].set_index("clip")[metric].dropna()
                b = df[df.method == m].set_index("clip")[metric].reindex(a.index).dropna()
                a = a.reindex(b.index)
                if len(a) > 5 and np.any(a.values != b.values):
                    st, p = wilcoxon(a, b)
                    out.append(dict(vs=m, metric=metric, n=len(a), mean_ours=a.mean(), mean_other=b.mean(), p=p))
        pd.DataFrame(out).to_csv(os.path.join(args.out, "wilcoxon.csv"), index=False)
        print(pd.DataFrame(out).round(4).to_string())
    except ImportError:
        print("scipy не знайдено: пропущено тест Вілкоксона")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for metric, yl in (("path_err_px", "Похибка траєкторії, px"), ("jitter_px", "Jitter (RMS 2-ї різниці), px")):
            pv = df.pivot_table(index="category", columns="method", values=metric)
            ax = pv.plot.bar(figsize=(12, 4.5)); ax.set_ylabel(yl); ax.set_xlabel("")
            plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            plt.savefig(os.path.join(args.out, "figures", f"{metric}.png"), dpi=160); plt.close()
    except ImportError:
        print("matplotlib не знайдено: пропущено графіки")


if __name__ == "__main__":
    main()
