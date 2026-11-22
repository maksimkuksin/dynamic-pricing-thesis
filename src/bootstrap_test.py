# -*- coding: utf-8 -*-
"""Парный бутстреп-тест разности WMAPE модели и базовых прогнозов.

Ресэмплирование по товарам (парное: в каждой повторной выборке сравниваются
те же товары), 10 000 повторных выборок, фиксированный seed.
Результат: reports/бутстреп_тест.json
"""
import json

import numpy as np
import pandas as pd

from prepare_data import ROOT

N_BOOTSTRAP = 10_000
SEED = 42


def main():
    df = pd.read_pickle(ROOT / "data" / "test_predictions.pkl")
    y = df["количество"].to_numpy(float)
    err = {
        "модель": np.abs(y - df["прогноз"].to_numpy(float)),
        "наивный_прогноз": np.abs(y - df["продажи_лаг1"].to_numpy(float)),
        "скользящее_среднее": np.abs(y - df["продажи_скольз4"].to_numpy(float)),
    }
    _, inv = np.unique(df["Наименование"].to_numpy(), return_inverse=True)
    n = inv.max() + 1

    # по-товарные суммы ошибок и фактических продаж (веса WMAPE)
    sums, w = {}, np.zeros(n)
    np.add.at(w, inv, y)
    for name, e in err.items():
        s = np.zeros(n)
        np.add.at(s, inv, e)
        sums[name] = s

    rng = np.random.default_rng(SEED)
    out = {"описание": "парный бутстреп по товарам разности WMAPE",
           "повторных_выборок": N_BOOTSTRAP, "seed": SEED,
           "разности_пп": {}}
    for base in ("наивный_прогноз", "скользящее_среднее"):
        d = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            ww = w[idx].sum()
            d[b] = sums["модель"][idx].sum() / ww * 100 \
                 - sums[base][idx].sum() / ww * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["разности_пп"][base] = {
            "точечная": round(float(d.mean()), 2),
            "ди_95": [round(float(lo), 1), round(float(hi), 1)],
            "p_value": float((d >= 0).mean()),
        }
        print(base, out["разности_пп"][base])

    path = ROOT / "reports" / "бутстреп_тест.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("записано:", path)


if __name__ == "__main__":
    main()
