# -*- coding: utf-8 -*-
"""Анализ чувствительности экономического эффекта.

Оценка эффекта в optimize_price.py — модельная: цена выбирается по argmax
ожидаемой прибыли и оценивается той же моделью («проклятие победителя»).
Здесь проверяется устойчивость вывода к допущениям:

 1. Ширина ценовой сетки: ±5 %, ±10 %, ±15 % (шаг 1 %).
 2. Усиленная эластичность: отклонение прогнозного спроса от базового
    возводится в квадрат, q' = q0 * (q / q0)^2, — имитация вдвое более
    сильной реакции спроса на цену, чем выучила модель.
 3. Бутстрэп по товарам (1000 ресэмплов с возвращением): доверительный
    интервал эффекта за счёт состава ассортимента.

Выход: reports/анализ_чувствительности.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_price import (MIN_MARGIN, STEP, load, set_price,  # noqa: E402
                            expected_profit)

N_BOOTSTRAP = 1000
SEED = 42


def optimize_for_grid(model, features, test, max_change, demand_scale=1.0):
    """Эффект рекомендаций для сетки ±max_change; demand_scale=2 — квадрат
    отклонения спроса (удвоенная эластичность)."""
    grid = np.arange(1 - max_change, 1 + max_change + 1e-9, STEP)
    fact_price = test["цена_продажи"].to_numpy()
    cost = test["цена_закупки"].to_numpy()
    n, k = len(test), len(grid)

    cand = (fact_price[:, None] * grid[None, :]).ravel()
    expanded = test.loc[np.repeat(test.index, k)].reset_index(drop=True)
    rows = set_price(expanded, cand)
    q = np.expm1(model.predict(rows[features])).clip(min=0)

    if demand_scale != 1.0:
        q0 = np.repeat(
            np.expm1(model.predict(set_price(test, fact_price)[features]))
            .clip(min=0), k)
        q = q0 * (q / np.maximum(q0, 1e-9)) ** demand_scale

    profit = (cand - np.repeat(cost, k)) * q
    margin = (cand - np.repeat(cost, k)) / cand
    profit[margin < MIN_MARGIN] = -np.inf

    profit2d = profit.reshape(n, k)
    all_bad = ~np.isfinite(profit2d).any(axis=1)
    best = profit2d.argmax(axis=1)
    rec_profit = profit2d[np.arange(n), best]

    fact_profit, _ = expected_profit(model, features, test, fact_price)
    rec_profit[all_bad] = fact_profit[all_bad]
    return fact_profit, rec_profit


def main() -> None:
    model, features, test = load()
    test = test.reset_index(drop=True)

    out = {}

    # --- 1. ширина сетки --------------------------------------------------
    grids = {}
    base = None
    for mc in (0.05, 0.10, 0.15):
        fp, rp = optimize_for_grid(model, features, test, mc)
        eff = float(rp.sum() / fp.sum() - 1) * 100
        grids[f"±{int(mc * 100)} %"] = round(eff, 2)
        if abs(mc - 0.15) < 1e-9:
            base = (fp, rp)
    out["сетки"] = grids

    # --- 2. удвоенная эластичность ----------------------------------------
    fp2, rp2 = optimize_for_grid(model, features, test, 0.15,
                                 demand_scale=2.0)
    out["эластичность_x2"] = {
        "эффект_проц": round(float(rp2.sum() / fp2.sum() - 1) * 100, 2),
        "описание": "отклонение прогноза спроса от базового возведено "
                    "в квадрат (вдвое более сильная реакция на цену)",
    }

    # --- 3. бутстрэп по товарам -------------------------------------------
    fp, rp = base
    products = test["Наименование"].to_numpy()
    uniq, inv = np.unique(products, return_inverse=True)
    rng = np.random.default_rng(SEED)
    effects = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(uniq), len(uniq))
        counts = np.bincount(sample, minlength=len(uniq))[inv]
        effects[b] = (rp * counts).sum() / (fp * counts).sum() - 1
    out["бутстрэп_по_товарам"] = {
        "ресэмплов": N_BOOTSTRAP,
        "эффект_медиана_проц": round(float(np.median(effects) * 100), 2),
        "ди_95_проц": [round(float(x) * 100, 2)
                       for x in np.percentile(effects, [2.5, 97.5])],
        "мин_проц": round(float(effects.min() * 100), 2),
        "макс_проц": round(float(effects.max() * 100), 2),
    }

    (REPORTS / "анализ_чувствительности.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
