# -*- coding: utf-8 -*-
"""
Модуль формирования признаков и обучения модели прогноза спроса.

Вход:  Прототип/data/weekly_sales.pkl — агрегированные продажи «товар-неделя».
Выход: обученная модель LightGBM (pkl), метрики (json), важность признаков (csv).

Подход:
 - строится полная сетка «товар × неделя»: недели без продаж получают
   количество = 0 (это тоже информация о спросе);
 - цель — log1p(количество): сглаживает редкие оптовые всплески;
 - признаки: цена и её относительные показатели, акция, сезонность,
   лаги и скользящие средние продаж, категория;
 - проверка — разделение по времени: последние 8 недель — тест;
 - базовые варианты для сравнения: наивный прогноз (продажи прошлой
   недели) и скользящее среднее за 4 недели.
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "weekly_sales.pkl"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TEST_WEEKS = 8          # последние 8 недель — проверочный период
LAGS = (1, 2, 4)        # лаги продаж
ROLL = 4                # окно скользящих средних


def build_grid(w: pd.DataFrame) -> pd.DataFrame:
    """Полная сетка товар × неделя; нулевые продажи восстановлены."""
    weeks = w[["дата_недели", "год", "неделя"]].drop_duplicates()
    grid = (w[["Наименование"]].drop_duplicates()
            .merge(weeks, how="cross"))
    g = grid.merge(w, on=["Наименование", "дата_недели", "год", "неделя"],
                   how="left")
    g["Категория"] = g.groupby("Наименование")["Категория"].ffill().bfill()
    g["количество"] = g["количество"].fillna(0)
    g["доля_акций"] = g["доля_акций"].fillna(0)
    # цена в неделю без продаж = последняя известная цена товара
    g = g.sort_values(["Наименование", "дата_недели"])
    for col in ("цена_продажи", "цена_закупки"):
        g[col] = g.groupby("Наименование")[col].ffill()
    return g


def add_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["Наименование", "дата_недели"]).copy()
    grp = g.groupby("Наименование")

    for lag in LAGS:
        g[f"продажи_лаг{lag}"] = grp["количество"].shift(lag)
    g["продажи_скольз4"] = grp["количество"].transform(
        lambda x: x.shift(1).rolling(ROLL, min_periods=1).mean())
    g["цена_лаг1"] = grp["цена_продажи"].shift(1)
    g["изменение_цены_проц"] = (g["цена_продажи"] / g["цена_лаг1"] - 1) * 100

    # относительная цена: к скользящему среднему цены категории
    # за 4 ПРОШЛЫЕ недели (shift(1) — текущая неделя не участвует,
    # чтобы исключить утечку информации проверочного периода)
    cat_week = (g.groupby(["Категория", "дата_недели"])["цена_продажи"]
                .mean().rename("кат_цена").reset_index()
                .sort_values(["Категория", "дата_недели"]))
    cat_week["кат_сред_4н"] = (cat_week.groupby("Категория")["кат_цена"]
                               .transform(lambda x: x.shift(1)
                                          .rolling(ROLL, min_periods=1)
                                          .mean()))
    g = g.merge(cat_week[["Категория", "дата_недели", "кат_сред_4н"]],
                on=["Категория", "дата_недели"], how="left")
    g["цена_к_средней_категории"] = g["цена_продажи"] / g["кат_сред_4н"]

    # сезонность
    g["неделя_sin"] = np.sin(2 * np.pi * g["неделя"] / 52)
    g["неделя_cos"] = np.cos(2 * np.pi * g["неделя"] / 52)
    g["месяц"] = g["дата_недели"].dt.month
    g["акция"] = (g["доля_акций"] > 0.5).astype(int)
    g["маржа_проц"] = ((g["цена_продажи"] - g["цена_закупки"])
                       / g["цена_продажи"] * 100)
    return g


FEATURES = ["цена_продажи", "изменение_цены_проц", "цена_к_средней_категории",
            "акция", "маржа_проц", "продажи_лаг1", "продажи_лаг2",
            "продажи_лаг4", "продажи_скольз4", "неделя_sin", "неделя_cos",
            "месяц", "Категория"]


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1e-9) * 100)


def main() -> None:
    w = pd.read_pickle(DATA)
    g = add_features(build_grid(w))
    g = g.dropna(subset=[f"продажи_лаг{LAGS[-1]}", "цена_лаг1"])
    g["Категория"] = g["Категория"].astype("category")

    cutoff = g["дата_недели"].max() - pd.Timedelta(weeks=TEST_WEEKS)
    train = g[g["дата_недели"] <= cutoff]
    test = g[g["дата_недели"] > cutoff]

    y_train = np.log1p(train["количество"])
    y_test = test["количество"].to_numpy()

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], y_train,
              categorical_feature=["Категория"])

    pred = np.expm1(model.predict(test[FEATURES])).clip(min=0)

    # базовые варианты
    naive = test["продажи_лаг1"].to_numpy()
    roll = test["продажи_скольз4"].to_numpy()

    metrics = {
        "период_обучения": f'{train["дата_недели"].min():%d.%m.%Y} — '
                           f'{train["дата_недели"].max():%d.%m.%Y}',
        "период_проверки": f'{test["дата_недели"].min():%d.%m.%Y} — '
                           f'{test["дата_недели"].max():%d.%m.%Y}',
        "строк_обучение": len(train), "строк_проверка": len(test),
        "модель": {"MAE": round(float(np.abs(y_test - pred).mean()), 3),
                   "WMAPE_проц": round(wmape(y_test, pred), 2)},
        "наивный_прогноз": {
            "MAE": round(float(np.abs(y_test - naive).mean()), 3),
            "WMAPE_проц": round(wmape(y_test, naive), 2)},
        "скользящее_среднее": {
            "MAE": round(float(np.abs(y_test - roll).mean()), 3),
            "WMAPE_проц": round(wmape(y_test, roll), 2)},
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "demand_lgbm.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)
    (REPORTS / "метрики_модели.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    imp = (pd.DataFrame({"признак": FEATURES,
                         "важность": model.feature_importances_})
           .sort_values("важность", ascending=False))
    imp.to_csv(REPORTS / "важность_признаков.csv", index=False,
               encoding="utf-8-sig")

    test.assign(прогноз=pred).to_pickle(DATA.parent / "test_predictions.pkl")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nВажность признаков:")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
