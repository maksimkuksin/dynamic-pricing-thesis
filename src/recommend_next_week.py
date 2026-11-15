# -*- coding: utf-8 -*-
"""Рекомендации цен на следующую неделю (боевой режим работы системы).

В отличие от optimize_price.py, который оценивает эффект на историческом
проверочном периоде, этот модуль строит рекомендации на будущую неделю —
первую неделю после конца данных. Именно такой отчёт категорийный
менеджер получал бы от системы на практике.

Схема:
 - к недельной сетке дописывается одна будущая неделя на каждый активный
   товар (продавался хотя бы раз за последние 8 недель данных);
 - цена и себестоимость будущей недели — последние известные значения
   товара (это базовый сценарий «оставить цену как есть»);
 - признаки (лаги, скользящие, относительные цены) считаются той же
   функцией add_features, что и при обучении, — без утечки: все
   скользящие статистики смотрят только в прошлое;
 - для каждой позиции перебирается сетка цен ±15 % (шаг 1 %) с
   ограничением минимальной маржи 10 % — как в optimize_price.py;
 - вывод: рекомендованная цена, прогноз спроса и ожидаемая прибыль,
   сравнение со сценарием «оставить текущую цену».

Выход: reports/рекомендации_следующая_неделя.csv
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_price import (MAX_CHANGE, MIN_MARGIN, STEP,  # noqa: E402
                            expected_profit)
from train_model import FEATURES, add_features, build_grid  # noqa: E402

ACTIVE_WEEKS = 8   # товар «активен», если продавался за последние 8 недель


def main() -> None:
    model = lgb.Booster(model_str=(ROOT / "models" / "demand_lgbm.txt")
                        .read_text(encoding="utf-8"))
    features = json.loads(
        (ROOT / "models" / "demand_lgbm_features.json")
        .read_text(encoding="utf-8"))

    w = pd.read_pickle(DATA / "weekly_sales.pkl")
    g = build_grid(w)

    last_week = w["дата_недели"].max()
    next_week = last_week + pd.Timedelta(days=7)
    iso = next_week.isocalendar()

    # активный ассортимент: продажи за последние ACTIVE_WEEKS недель
    recent = w[w["дата_недели"] > last_week
               - pd.Timedelta(weeks=ACTIVE_WEEKS)]
    active = (recent[recent["количество"] > 0]["Наименование"].unique())

    last_rows = (g.sort_values("дата_недели")
                 .groupby("Наименование").tail(1)
                 .set_index("Наименование"))
    fut = pd.DataFrame({
        "Наименование": active,
        "дата_недели": next_week,
        "год": iso.year,
        "неделя": iso.week,
        "Категория": last_rows.loc[active, "Категория"].to_numpy(),
        "количество": 0,
        "цена_продажи": last_rows.loc[active, "цена_продажи"].to_numpy(),
        "цена_закупки": last_rows.loc[active, "цена_закупки"].to_numpy(),
    })
    g = pd.concat([g, fut], ignore_index=True)
    g = add_features(g)

    nxt = g[g["дата_недели"] == next_week].dropna(subset=features)
    # категориальный dtype должен совпадать с обучением: берём его из
    # сохранённого проверочного набора
    cat_dtype = pd.read_pickle(DATA / "test_predictions.pkl")["Категория"].dtype
    nxt = nxt.copy()
    nxt["Категория"] = nxt["Категория"].astype(cat_dtype)
    nxt = nxt.reset_index(drop=True)

    # --- перебор сетки цен (та же логика, что в optimize_price.py) --------
    grid = np.arange(1 - MAX_CHANGE, 1 + MAX_CHANGE + 1e-9, STEP)
    cur_price = nxt["цена_продажи"].to_numpy()
    cost = nxt["цена_закупки"].to_numpy()
    n, k = len(nxt), len(grid)

    cand = (cur_price[:, None] * grid[None, :]).ravel()
    expanded = nxt.loc[np.repeat(nxt.index, k)].reset_index(drop=True)
    profit, demand = expected_profit(model, features, expanded, cand)

    margin = (cand - np.repeat(cost, k)) / cand
    profit[margin < MIN_MARGIN] = -np.inf

    profit2d = profit.reshape(n, k)
    demand2d = demand.reshape(n, k)
    all_bad = ~np.isfinite(profit2d).any(axis=1)
    best = profit2d.argmax(axis=1)
    rec_price = cand.reshape(n, k)[np.arange(n), best]
    rec_profit = profit2d[np.arange(n), best]
    rec_demand = demand2d[np.arange(n), best]

    keep_profit, keep_demand = expected_profit(model, features, nxt,
                                               cur_price)
    rec_price[all_bad] = cur_price[all_bad]
    rec_profit[all_bad] = keep_profit[all_bad]
    rec_demand[all_bad] = keep_demand[all_bad]

    change = (rec_price / cur_price - 1) * 100
    rec = pd.DataFrame({
        "Наименование": nxt["Наименование"],
        "Категория": nxt["Категория"].astype(str),
        "дата_недели": next_week.date().isoformat(),
        "цена_текущая": cur_price.round(2),
        "цена_рекомендованная": rec_price.round(2),
        "изменение_проц": change.round(1),
        "себестоимость": cost.round(2),
        "спрос_прогноз_шт": rec_demand.round(1),
        "прибыль_ожид_руб": rec_profit.round(0),
    })
    REPORTS.mkdir(parents=True, exist_ok=True)
    rec.to_csv(REPORTS / "рекомендации_следующая_неделя.csv",
               index=False, encoding="utf-8-sig")

    summary = {
        "неделя_рекомендаций": next_week.date().isoformat(),
        "активных_товаров": int(len(active)),
        "позиций_с_рекомендацией": int(len(rec)),
        "доля_изменений_цены_проц": round(float(
            (np.abs(change) > 0.5).mean() * 100), 1),
        "медиана_изменения_проц": round(float(np.median(change)), 1),
        "ожидаемая_прибыль_рекомендованная_руб":
            round(float(rec_profit.sum()), 0),
        "ожидаемая_прибыль_при_текущих_ценах_руб":
            round(float(keep_profit.sum()), 0),
        "модельный_эффект_проц": round(float(
            (rec_profit.sum() / keep_profit.sum() - 1) * 100), 2),
    }
    (REPORTS / "рекомендации_следующая_неделя.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nПримеры (топ-5 по ожидаемой прибыли):")
    print(rec.nlargest(5, "прибыль_ожид_руб").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
