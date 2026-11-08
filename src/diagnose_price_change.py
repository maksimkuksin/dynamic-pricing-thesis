# -*- coding: utf-8 -*-
"""
Диагностика артефакта признака «изменение_цены_проц».

Гипотеза рецензии: цена в недели без продаж протягивается вперёд (ffill),
поэтому изменение цены равно ровно 0 именно там, где продаж не было, —
признак косвенно кодирует целевую переменную.

Проверки:
 1. Табличная: доля нулевых продаж среди строк с изменением цены = 0
    против строк с ненулевым изменением (обучающий период).
 2. Кривая спроса: средний прогноз модели по сетке цен ±15 % с шагом 1 %
    для всех строк проверочного периода. Если в точке «0 %» (цена прошлой
    недели) видна ступенька, артефакт подтверждён.

Выход: reports/диагностика_изменения_цены.json
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = lgb.Booster(model_str=(ROOT / "models" / "demand_lgbm.txt")
                        .read_text(encoding="utf-8"))
    features = json.loads(
        (ROOT / "models" / "demand_lgbm_features.json")
        .read_text(encoding="utf-8"))
    test = pd.read_pickle(ROOT / "data" / "test_predictions.pkl")
    weekly = pd.read_pickle(ROOT / "data" / "weekly_sales.pkl")

    # --- проверка 1: нулевые продажи против нулевого изменения цены -------
    from train_model import build_grid, add_features
    g = add_features(build_grid(weekly))
    g = g.dropna(subset=["продажи_лаг4", "цена_лаг1"])
    cutoff = g["дата_недели"].max() - pd.Timedelta(weeks=8)
    train = g[g["дата_недели"] <= cutoff]

    zero_change = train["изменение_цены_проц"].abs() < 1e-9
    tab = {
        "строк_обучение": len(train),
        "доля_нулевого_изменения_проц": round(float(zero_change.mean() * 100), 1),
        "доля_нулевых_продаж_при_нулевом_изменении_проц": round(float(
            (train.loc[zero_change, "количество"] == 0).mean() * 100), 1),
        "доля_нулевых_продаж_при_ненулевом_изменении_проц": round(float(
            (train.loc[~zero_change, "количество"] == 0).mean() * 100), 1),
        "средние_продажи_при_нулевом_изменении": round(float(
            train.loc[zero_change, "количество"].mean()), 2),
        "средние_продажи_при_ненулевом_изменении": round(float(
            train.loc[~zero_change, "количество"].mean()), 2),
    }

    # --- проверка 2: средняя кривая спроса по сетке цен -------------------
    grid = np.arange(0.85, 1.1501, 0.01)
    n, k = len(test), len(grid)
    fact_price = test["цена_продажи"].to_numpy()
    cand = (fact_price[:, None] * grid[None, :]).ravel()
    exp = test.loc[np.repeat(test.index, k)].reset_index(drop=True)
    exp["цена_продажи"] = cand
    exp["изменение_цены_проц"] = (cand / exp["цена_лаг1"].to_numpy() - 1) * 100
    exp["цена_к_средней_категории"] = cand / exp["кат_сред_4н"].to_numpy()
    exp["маржа_проц"] = ((cand - exp["цена_закупки"].to_numpy()) / cand * 100)
    q = np.expm1(model.predict(exp[features])).clip(min=0)
    curve = q.reshape(n, k).mean(axis=0)

    # относительная кривая (к значению при фактической цене)
    i_fact = int(np.argmin(np.abs(grid - 1.0)))
    rel = curve / curve[i_fact] * 100

    # ступенька в 0 %: скачок между соседними точками вокруг цены прошлой
    # недели против типичного соседнего скачка
    diffs = np.diff(curve)
    i_zero = int(np.argmin(np.abs(grid - (test["цена_лаг1"] / fact_price)
                                          .mean())))
    step_at_zero = float(abs(diffs[max(i_zero - 1, 0)]))
    typical_step = float(np.median(np.abs(diffs)))
    max_step = float(np.max(np.abs(diffs)))
    i_max_step = int(np.argmax(np.abs(diffs)))

    result = {
        "проверка_1_табличная": tab,
        "проверка_2_кривая": {
            "сетка_множителей": [round(float(x), 2) for x in grid],
            "средний_прогноз_спроса": [round(float(x), 3) for x in curve],
            "относительно_фактической_цены_проц": [round(float(x), 1)
                                                   for x in rel],
            "индекс_точки_0_проц": i_zero,
            "множитель_в_точке_0_проц": round(float(grid[i_zero]), 3),
            "скачок_в_точке_0_проц": round(step_at_zero, 3),
            "типичный_соседний_скачок": round(typical_step, 3),
            "максимальный_скачок": round(max_step, 3),
            "множитель_максимального_скачка": round(float(grid[i_max_step]), 3),
        },
    }

    out = ROOT / "reports" / "диагностика_изменения_цены.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(result["проверка_1_табличная"], ensure_ascii=False,
                     indent=2))
    print("\nКривая (множитель -> средний прогноз, отн. к факту, %):")
    for x, c, r in zip(grid, curve, rel):
        marker = " <-- точка 0 %" if abs(x - grid[i_zero]) < 1e-9 else ""
        print(f"  {x:.2f} -> {c:8.3f}  ({r:6.1f} %){marker}")
    print(f"\nСкачок в точке 0 %: {step_at_zero:.3f}; "
          f"типичный соседний: {typical_step:.3f}; "
          f"максимум: {max_step:.3f} при x{grid[i_max_step]:.2f}")


if __name__ == "__main__":
    sys.exit(main())
