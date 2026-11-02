#!/usr/bin/env python3
"""Збирає набір даних на диск (mp4 + еталонна траєкторія .npz).

  python3 make_dataset.py --out dataset --seeds 0 1 2 3 4 5 --T 150            # синтетика: 10 категорій
  python3 make_dataset.py --out dataset --stable_videos my_stable/ --T 300       # + ваші стабільні відео з доданим тремтінням

Баланс: однакова кількість кліпів у кожній категорії. 10 категорій × 6 seed × 150 кадрів = 9000 кадрів (≥1000).
Для кожного кліпу зберігаються: <name>_shaky.mp4, <name>_ideal.mp4 (еталон), <name>_gt.npz (P, J).
"""
import argparse, glob, os, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidstab.synth import CATEGORIES, render_clip, shake_real_video
from run_experiments import read_video


def save(frames, path, fps=30):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames: wr.write(f)
    wr.release()


def dump(out, name, sh, ideal, gt):
    save(sh, os.path.join(out, name + "_shaky.mp4")); save(ideal, os.path.join(out, name + "_ideal.mp4"))
    np.savez(os.path.join(out, name + "_gt.npz"), P=gt["P"], J=gt["J"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--T", type=int, default=150); ap.add_argument("--stable_videos")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); total = 0; clips = 0
    for cat in CATEGORIES:
        for s in a.seeds:
            sh, ideal, gt = render_clip(cat, s, T=a.T); dump(a.out, f"{cat}_{s}", sh, ideal, gt)
            total += len(sh); clips += 1
    if a.stable_videos:
        for p in sorted(glob.glob(os.path.join(a.stable_videos, "*"))):
            base = read_video(p, a.T)
            for kind in ("low", "medium", "high", "walking"):
                sh, ideal, gt = shake_real_video(base, kind, seed=1)
                dump(a.out, f"real_{os.path.splitext(os.path.basename(p))[0]}_{kind}", sh, ideal, gt)
                total += len(sh); clips += 1
    print(f"Збережено {clips} кліпів, {total} кадрів у {a.out}/")


if __name__ == "__main__":
    main()
