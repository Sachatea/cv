# Компенсація тряски кадру (онлайн-стабілізація відео) — Лабораторна, тема 15

Каузальний (без затримки) стабілізатор: кадр на вході → кадр, скомпенсований відносно попереднього, на виході.

## Запуск на macOS (M1)
```bash
cd cv
/usr/local/bin/python3 -m pip install -r requirements.txt
# 1) інтерактивне демо (вебкамера; при першому запуску дозвольте доступ до камери для Terminal)
/usr/local/bin/python3 demo.py
/usr/local/bin/python3 demo.py --source мій_відос.mp4
# 2) експерименти (≈20–30 хв): CSV + графіки + тест Вілкоксона у results/
/usr/local/bin/python3 run_experiments.py --seeds 200 201 202 203 --T 150
# 3) з вашими відео:
#    stable/  — стабільні відео (штатив) → до них додається відомий тремор, еталон відомий
#    shaky/   — реальні «тремтячі» відео (ходьба, в руках) → метрики без еталона (ITF, jitter-ratio)
/usr/local/bin/python3 run_experiments.py --stable_videos stable --shaky_videos shaky --T 300 --out results_real
# 4) зберегти набір даних на диск (≥1000 елементів)
/usr/local/bin/python3 make_dataset.py --out dataset --stable_videos stable
```
Якщо `python3 -m pip` скаржиться на "externally-managed", додайте `--break-system-packages` або створіть venv.

## Структура
- `vidstab/motion.py` — оцінка руху: Shi–Tomasi (сітка 6×8), піраміальний Lucas–Kanade + forward–backward, RANSAC-подібність, критерії надійності, детекція склейки.
- `vidstab/filters.py` — каузальні згладжувачі траєкторії: MA, Калман (CV-модель), **Intent-Калман** (адаптивний q за статистикою інновацій); гауссів некаузальний еталон.
- `vidstab/stabilizer.py` — онлайн-конвеєр + фіксований зум за найгіршим запасом (без чорних країв).
- `vidstab/synth.py`, `metrics.py`, `evaluate.py` — синтетичні кліпи з відомим тремтінням (5 звичайних + 5 worst-case категорій), метрики.
- `run_experiments.py`, `make_dataset.py`, `demo.py`, `tools/` (підбір гіперпараметрів на окремих seed 100, 101).

## Клавіші демо
SPACE увімк/вимк · M метод · P пауза · R скинути · S знімок · Q вихід. Повзунки: Smooth, Crop %, Compare.

## Примітка про результати
`results_sandbox/` — перший прогін (старі типові налаштування: Intent + gating). За ним і за прогоном на реальних відео:
адаптивний Intent не кращий за звичайний Калман, а gating помилково вважав швидкий розмитий рух «склейкою» (кліп shaky_1, кадри 208–222).
Тому типовий метод тепер **Kalman без gating** (`StabConfig()`); решта варіантів лишилась як абляції у `run_experiments.py`.
Оскільки цей вибір зроблено за тими ж даними, фінальне порівняння слід запускати на НОВИХ seed (300–303):
`python3 run_experiments.py --seeds 300 301 302 303 --stable_videos stable_v --shaky_videos shaky_v --T 300 --out results_final`
