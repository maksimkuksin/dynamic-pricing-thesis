# -*- coding: utf-8 -*-
"""
Модуль формирования признаков и обучения модели прогноза спроса.

Вход:  Прототип/data/weekly_sales.pkl — агрегированные продажи «товар-неделя».
Выход: обученная модель LightGBM (нативный формат txt + список признаков
       в json), метрики (json), важность признаков (csv).

Подход:
 - строится сетка «товар × неделя» внутри периода доступности товара:
   недели без продаж получают количество = 0 (это тоже информация
   о спросе), недели до появления товара в ассортименте исключаются;
 - цель — log1p(количество): сглаживает редкие оптовые всплески;
 - признаки: цена и её относительные показатели, сезонность,
   лаги и скользящие средние продаж, категория;
 - проверка — разделение по времени: последние 8 недель — тест;
 - базовые варианты для сравнения: наивный прогноз (продажи прошлой
   недели), скользящее среднее за 4 недели, случайный лес и
   лог-лог регрессия (Ridge) на том же разделении.

Примечание по признакам: «изменение_цены_проц» сознательно исключён
из обучения. Цена в недели без продаж протягивается вперёд, поэтому
нулевое изменение цены почти всегда означает нулевые продажи — признак
косвенно кодировал целевую переменную (см. src/diagnose_price_change.py,
reports/диагностика_изменения_цены.json). Столбец по-прежнему считается
и сохраняется в данных для контрфактических проверок.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "weekly_sales.pkl"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TEST_WEEKS = 8          # последние 8 недель — проверочный период
LAGS = (1, 2, 4)        # лаги продаж
ROLL = 4                # окно скользящих средних


def build_grid(w: pd.DataFrame) -> pd.DataFrame:
    """Сетка товар × неделя; нулевые продажи восстановлены.

    Сетка строится только внутри периода реальной доступности товара:
    недели до первой продажи не включаются, чтобы не порождать
    искусственные нулевые продажи для ещё не заведённого ассортимента.
    """
    weeks = w[["дата_недели", "год", "неделя"]].drop_duplicates()
    first = (w.groupby("Наименование")["дата_недели"].min()
             .rename("первая_неделя"))
    grid = (w[["Наименование"]].drop_duplicates()
            .merge(weeks, how="cross")
            .merge(first, on="Наименование"))
    grid = grid[grid["дата_недели"] >= grid["первая_неделя"]]
    g = grid.merge(w, on=["Наименование", "дата_недели", "год", "неделя"],
                   how="left")
    g["Категория"] = g.groupby("Наименование")["Категория"].ffill().bfill()
    g["количество"] = g["количество"].fillna(0)
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
    g["маржа_проц"] = ((g["цена_продажи"] - g["цена_закупки"])
                       / g["цена_продажи"] * 100)
    return g


FEATURES = ["цена_продажи", "цена_к_средней_категории",
            "маржа_проц", "продажи_лаг1", "продажи_лаг2",
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
    t0 = time.perf_counter()
    model.fit(train[FEATURES], y_train,
              categorical_feature=["Категория"])
    t_lgbm = time.perf_counter() - t0

    pred = np.expm1(model.predict(test[FEATURES])).clip(min=0)

    # базовые варианты
    naive = test["продажи_лаг1"].to_numpy()
    roll = test["продажи_скольз4"].to_numpy()

    # сравнительные модели на том же разделении: категория — one-hot,
    # для лог-лог регрессии цена берётся в логарифме (цель уже log1p)
    Xtr = pd.get_dummies(train[FEATURES], columns=["Категория"])
    Xte = pd.get_dummies(test[FEATURES], columns=["Категория"])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    rf.fit(Xtr, y_train)
    t_rf = time.perf_counter() - t0
    pred_rf = np.expm1(rf.predict(Xte)).clip(min=0)

    Xtr_log = Xtr.assign(цена_продажи=np.log(Xtr["цена_продажи"]))
    Xte_log = Xte.assign(цена_продажи=np.log(Xte["цена_продажи"]))
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr_log, y_train)
    pred_ridge = np.expm1(ridge.predict(Xte_log)).clip(min=0)

    def score(y_pred: np.ndarray) -> dict:
        return {"MAE": round(float(np.abs(y_test - y_pred).mean()), 3),
                "WMAPE_проц": round(wmape(y_test, y_pred), 2)}

    metrics = {
        "период_обучения": f'{train["дата_недели"].min():%d.%m.%Y} — '
                           f'{train["дата_недели"].max():%d.%m.%Y}',
        "период_проверки": f'{test["дата_недели"].min():%d.%m.%Y} — '
                           f'{test["дата_недели"].max():%d.%m.%Y}',
        "строк_обучение": len(train), "строк_проверка": len(test),
        "время_обучения_сек": {"LightGBM": round(t_lgbm, 2),
                               "случайный_лес": round(t_rf, 2)},
        "модель": score(pred),
        "наивный_прогноз": score(naive),
        "скользящее_среднее": score(roll),
        "случайный_лес": score(pred_rf),
        "лог_лог_регрессия": score(pred_ridge),
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # модель — в нативном текстовом формате LightGBM (воспроизводимо
    # между версиями python/pickle), список признаков — рядом в json.
    # LightGBM C API не принимает не-ASCII пути, поэтому сохраняем
    # во временный файл и копируем средствами Python.
    import shutil
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "demand_lgbm.txt"
    model.booster_.save_model(str(tmp))
    shutil.copyfile(tmp, MODELS / "demand_lgbm.txt")
    (MODELS / "demand_lgbm_features.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2), encoding="utf-8")
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
