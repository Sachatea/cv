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
/usr/local/bin/python3 run_experiments.py --seeds 400 401 402 403 404 405 406 407 --T 300
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
- `vidstab/stabilizer.py` — онлайн-конвеєр + адаптивний запас κ_t і зум Z(κ_t) за таблицею required_zoom (без чорних країв); ECC-запас і м'яка вага вимірювань налаштовуються в `StabConfig`.
- `vidstab/synth.py`, `metrics.py`, `evaluate.py` — синтетичні кліпи з відомим тремтінням (5 звичайних + 5 worst-case категорій), метрики.
- `run_experiments.py`, `make_dataset.py`, `demo.py`, `tools/` (підбір гіперпараметрів на окремих seed 100, 101).

## Клавіші демо
SPACE увімк/вимк · M метод · A адаптивний/фіксований запас · P пауза · R скинути · S знімок · Q вихід. Повзунки: Smooth, Crop % (лише для фіксованого запасу), Compare.

## Що нового (v3)
Фінальний метод (`StabConfig()` за замовчуванням) = Калман + **адаптивний запас/зум** (κ повільно змінюється за амплітудою тремтіння) +
**ECC-запасний оцінювач** для слабкотекстурних сцен + **м'яка вага вимірювань** (шум вимірювання залежить від якості оцінки руху).
Адаптивний Intent і gating залишено лише як абляції раніше (не допомогли). Клавіша **A** у демо — перемикач адаптивного/фіксованого запасу.
Фінальний прогін на нових seed (400–407): див. команду вище (потрібно ≈ 30–40 хв).

## Додаткові аналізи (tools/) — використані у звіті
```bash
python3 tools/analyze_results.py results_v3/per_clip.csv analysis.json   # iso-area, парні тести, bootstrap-ДІ
python3 tools/mixed_experiment.py       # нестаціонарне тремтіння (пісочничний експеримент, ≈10 хв)
python3 tools/matched_filters.py        # Калман проти MA-9 за однакового запасу (≈10 хв)
python3 tools/noise_sensitivity.py      # твердження 6: чутливість до шуму приростів
python3 tools/coverage_check.py [video.mp4]   # геометрична перевірка відсутності країв
```
Фінальна версія звіту: частина експериментів (mixed, matched_filters, noise_sensitivity, coverage_check, рис. 2, 5, 9а, 10, 11) виконана в пісочниці (Linux, 2 ядра);
основні таблиці (3, 4, 7–9) — на Apple M1 (`results_v3`).
`border` у per_clip.csv не інформативна (режим BORDER_REPLICATE) — див. звіт, твердження 3.
