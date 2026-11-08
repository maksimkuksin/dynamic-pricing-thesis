# -*- coding: utf-8 -*-
"""
Модуль оптимизации цены и оценки экономического эффекта.

Вход:  модель спроса (models/demand_lgbm.pkl), прогнозы тестового периода
       (data/test_predictions.pkl), недельные данные (data/weekly_sales.pkl).
Выход: рекомендации цен (csv), сводка эксперимента (json).

Логика:
 - для каждой пары «товар-неделя» проверочного периода перебирается сетка
   допустимых цен (±15% от фактической, шаг 1%);
 - для каждой цены-кандидата пересчитываются ценозависимые признаки,
   модель прогнозирует спрос, считается ожидаемая прибыль
   (цена − себестоимость) × прогнозный спрос;
 - рекомендуется цена с максимальной прибылью при ограничениях:
   маржа не ниже 10%, изменение цены не более 15%;
 - эффект оценивается сравнением ожидаемой прибыли трёх стратегий:
   фактическая цена, «себестоимость + 30%», рекомендованная цена.

Ограничение (для текста ВКР): спрос при нереализованных ценах оценивается
моделью, построенной на наблюдаемых данных, — это модельный эксперимент,
а не результат реального A/B-теста.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

MIN_MARGIN = 0.10      # минимальная маржа 10%
MAX_CHANGE = 0.15      # максимальное изменение цены ±15%
STEP = 0.01            # шаг сетки 1%
MARKUP_BASELINE = 0.30  # базовая стратегия «себестоимость + 30%»


def load() -> tuple:
    with open(ROOT / "models" / "demand_lgbm.pkl", "rb") as f:
        bundle = pickle.load(f)
    test = pd.read_pickle(DATA / "test_predictions.pkl")
    return bundle["model"], bundle["features"], test


def set_price(df: pd.DataFrame, price: np.ndarray) -> pd.DataFrame:
    """Пересчёт ценозависимых признаков под заданную цену."""
    out = df.copy()
    out["цена_продажи"] = price
    out["изменение_цены_проц"] = (price / out["цена_лаг1"] - 1) * 100
    cat_mean = df["цена_продажи"] / df["цена_к_средней_категории"]
    out["цена_к_средней_категории"] = price / cat_mean
    out["маржа_проц"] = (price - out["цена_закупки"]) / price * 100
    return out


def expected_profit(model, features, base: pd.DataFrame,
                    price: np.ndarray) -> np.ndarray:
    """Ожидаемая прибыль стратегии = (цена − себестоимость) × прогноз спроса."""
    rows = set_price(base, price)
    q = np.expm1(model.predict(rows[features])).clip(min=0)
    return (price - base["цена_закупки"].to_numpy()) * q, q


def main() -> None:
    model, features, test = load()
    test = test.reset_index(drop=True)

    # --- перебор сетки цен ------------------------------------------------
    grid = np.arange(1 - MAX_CHANGE, 1 + MAX_CHANGE + 1e-9, STEP)
    fact_price = test["цена_продажи"].to_numpy()
    cost = test["цена_закупки"].to_numpy()

    n, k = len(test), len(grid)
    cand = (fact_price[:, None] * grid[None, :]).ravel()          # (n*k,)
    expanded = test.loc[np.repeat(test.index, k)].reset_index(drop=True)

    profit, demand = expected_profit(model, features, expanded, cand)

    # запретные кандидаты: маржа ниже порога
    margin = (cand - np.repeat(cost, k)) / cand
    profit[margin < MIN_MARGIN] = -np.inf

    profit2d = profit.reshape(n, k)
    demand2d = demand.reshape(n, k)

    # товары, где все кандидаты запрещены (себестоимость почти равна цене):
    # рекомендация = сохранить фактическую цену
    all_bad = ~np.isfinite(profit2d).any(axis=1)
    best = profit2d.argmax(axis=1)
    rec_price = cand.reshape(n, k)[np.arange(n), best]
    rec_profit = profit2d[np.arange(n), best]
    rec_demand = demand2d[np.arange(n), best]

    fact_profit_model, fact_demand_model = expected_profit(
        model, features, test, fact_price)
    rec_price[all_bad] = fact_price[all_bad]
    rec_profit[all_bad] = fact_profit_model[all_bad]
    rec_demand[all_bad] = fact_demand_model[all_bad]
    cp_price = cost * (1 + MARKUP_BASELINE)                        # cost+30%
    cp_profit, cp_demand = expected_profit(model, features, test, cp_price)

    # --- оценка эластичности спроса по модели -----------------------------
    q_up, q_down = demand2d[:, -1], demand2d[:, 0]  # ±15% от фактической цены
    with np.errstate(divide="ignore", invalid="ignore"):
        elast = ((q_up - q_down) / np.maximum(q_down, 1e-9)) / 0.30
    elast = elast[np.isfinite(elast)]

    # фактически реализованная прибыль (по данным продаж)
    fact_profit_real = ((test["цена_продажи"] - test["цена_закупки"])
                        * test["количество"]).sum()

    change = (rec_price / fact_price - 1) * 100
    summary = {
        "строк_товар_неделя": n,
        "ожидаемая_прибыль": {
            "фактическая_цена": round(float(fact_profit_model.sum()), 0),
            "себестоимость_плюс_30проц": round(float(cp_profit.sum()), 0),
            "рекомендованная_цена": round(float(rec_profit.sum()), 0),
        },
        "эффект_к_фактической_цене_проц": round(float(
            (rec_profit.sum() / fact_profit_model.sum() - 1) * 100), 2),
        "эффект_к_cost_plus_проц": round(float(
            (rec_profit.sum() / cp_profit.sum() - 1) * 100), 2),
        "фактически_реализованная_прибыль_за_период": round(
            float(fact_profit_real), 0),
        "рекомендации": {
            "доля_изменений_цены_проц": round(float(
                (np.abs(change) > 0.5).mean() * 100), 1),
            "доля_повышений_проц": round(float((change > 0.5).mean() * 100), 1),
            "доля_понижений_проц": round(float((change < -0.5).mean() * 100), 1),
            "медиана_изменения_проц": round(float(np.median(change)), 2),
            "среднее_изменение_проц": round(float(change.mean()), 2),
        },
        "оценка_эластичности": {
            "медиана": round(float(np.median(elast)), 3),
            "доля_неэластичных_позиций_проц": round(float(
                (np.abs(elast) < 1).mean() * 100), 1),
        },
        "строк_без_изменения_из_за_ограничений": int(all_bad.sum()),
        "ограничения": {"мин_маржа": MIN_MARGIN,
                        "макс_изменение_цены": MAX_CHANGE, "шаг": STEP},
    }

    rec = pd.DataFrame({
        "Наименование": test["Наименование"], "Категория": test["Категория"],
        "дата_недели": test["дата_недели"],
        "цена_факт": fact_price.round(2), "цена_реком": rec_price.round(2),
        "изменение_проц": change.round(1),
        "себестоимость": cost.round(2),
        "спрос_прогноз": rec_demand.round(1),
        "прибыль_ожид": rec_profit.round(0),
    })
    REPORTS.mkdir(parents=True, exist_ok=True)
    rec.to_csv(REPORTS / "рекомендации_цен.csv", index=False,
               encoding="utf-8-sig")
    (REPORTS / "экономический_эффект.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nПримеры рекомендаций (топ-5 по прибыли):")
    print(rec.nlargest(5, "прибыль_ожид").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
