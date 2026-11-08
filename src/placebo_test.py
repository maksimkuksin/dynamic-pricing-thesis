# -*- coding: utf-8 -*-
"""Плацебо-тест (диагностический отрицательный контроль): что даёт
модель, обученная на данных, где согласованная во времени связь
цены и продаж разрушена?

Идея (по замечанию внешней проверки): если экономический эффект
оптимизации цен воспроизводится моделью, обученной на перемешанных
по времени ценах, то эффект определяется не ценовой реакцией, а
конструкцией задачи (прогноз держится на лагах продаж, а при слабом
отклике спроса рост цены повышает расчётную прибыль по большинству
позиций).

Схема:
 - сначала данные делятся по времени на обучающий и проверочный
   периоды (граница — первая неделя проверочного периода);
 - цены продажи перемешиваются по времени внутри каждого товара
   ТОЛЬКО в обучающем периоде (rng-перестановка): цены проверочного
   периода не перемешиваются и не могут попасть в обучение
   (временная утечка исключена); ценозависимые признаки (лаг цены,
   изменение цены, относительная цена, маржа) пересчитываются из
   перемешанной цены — исходная согласованная во времени связь цены
   и продаж разрушена (случайные корреляции при этом возможны),
   распределение цен товара в обучении сохранено;
 - проверочный период и все оценки (WMAPE, эффект оптимизации на
   сетке ±15 %, совпадение знака реакции) считаются на РЕАЛЬНЫХ
   данных — так же, как для основной модели;
 - прогонов три (разные seed), в отчёт пишутся значения каждого
   прогона и диапазон.

Интерпретация для текста ВКР: если плацебо-модель показывает эффект
того же порядка, что и реальная, то величина «+39,3 %» характеризует
результат оптимизации в пределах заданной ценовой сетки при слабом
модельном отклике, а не доказанный экономический эффект; выводы об
эффекте следует опирать на сценарный анализ с условными допущениями
о доле реализации модельного эффекта (10 / 25 / 50 %).

Выход: reports/плацебо_тест.json
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_price import MIN_MARGIN, STEP, expected_profit, set_price  # noqa: E402
from train_model import (FEATURES, LAGS, add_features, build_grid,  # noqa: E402
                         wmape)

N_RUNS = 3
MAX_CHANGE = 0.15   # основная сетка оптимизации
MIN_PRICE_CHANGE_PCT = 2.0  # как в validate_price_response.py


def shuffle_prices_train(g: pd.DataFrame, cutoff, seed: int) -> pd.DataFrame:
    """Перестановка цен по времени внутри каждого товара — только
    в обучающем периоде (строго раньше cutoff). Проверочный период
    не перемешивается и не участвует в перестановке, поэтому цены
    будущего не могут попасть в обучение."""
    rng = np.random.default_rng(seed)
    g = g.copy()
    mask = g["дата_недели"] < cutoff
    tr = g.loc[mask]
    g.loc[mask, "цена_продажи"] = (
        tr.groupby("Наименование")["цена_продажи"]
          .transform(lambda s: rng.permutation(s.to_numpy())))
    return g


def run(seed: int, w: pd.DataFrame, test: pd.DataFrame) -> dict:
    cutoff = test["дата_недели"].min()
    g = add_features(shuffle_prices_train(build_grid(w), cutoff, seed))
    g = g.dropna(subset=[f"продажи_лаг{LAGS[-1]}", "цена_лаг1"])
    g["Категория"] = g["Категория"].astype("category")
    train = g[g["дата_недели"] < cutoff]

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1)
    model.fit(train[FEATURES], np.log1p(train["количество"]),
              categorical_feature=["Категория"])

    # качество прогноза на реальном проверочном периоде
    pred = np.expm1(model.predict(test[FEATURES])).clip(min=0)
    wmape_pct = wmape(test["количество"].to_numpy(), pred)

    # важность признаков плацебо-модели: показывает, что «важность»
    # отражает использование признака, а не выученную реакцию на цену
    importance = dict(zip(FEATURES, model.feature_importances_.tolist()))

    # экономический эффект оптимизации на реальном тесте (сетка ±15 %)
    fact_price = test["цена_продажи"].to_numpy()
    cost = test["цена_закупки"].to_numpy()
    grid = np.arange(1 - MAX_CHANGE, 1 + MAX_CHANGE + 1e-9, STEP)
    n, k = len(test), len(grid)
    cand = (fact_price[:, None] * grid[None, :]).ravel()
    expanded = test.loc[np.repeat(test.index, k)].reset_index(drop=True)
    q = np.expm1(model.predict(set_price(expanded, cand)[FEATURES])) \
        .clip(min=0)
    profit = (cand - np.repeat(cost, k)) * q
    profit[(cand - np.repeat(cost, k)) / cand < MIN_MARGIN] = -np.inf
    profit2d = profit.reshape(n, k)
    all_bad = ~np.isfinite(profit2d).any(axis=1)
    best = profit2d.argmax(axis=1)
    rec_profit = profit2d[np.arange(n), best]
    fact_profit, _ = expected_profit(model, FEATURES, test, fact_price)
    rec_profit[all_bad] = fact_profit[all_bad]
    effect_pct = float(rec_profit.sum() / fact_profit.sum() - 1) * 100

    # совпадение знака реакции (та же схема, что в validate_price_response)
    moved = test[test["изменение_цены_проц"].abs()
                 >= MIN_PRICE_CHANGE_PCT].copy()
    cf = moved.copy()
    cf["цена_продажи"] = cf["цена_лаг1"]
    cf["цена_к_средней_категории"] = cf["цена_лаг1"] / cf["кат_сред_4н"]
    cf["маржа_проц"] = ((cf["цена_лаг1"] - cf["цена_закупки"])
                        / cf["цена_лаг1"] * 100)
    q_old = np.expm1(model.predict(cf[FEATURES])).clip(min=0)
    q_new = np.expm1(model.predict(moved[FEATURES])).clip(min=0)
    md, fd = q_new - q_old, (moved["количество"]
                             - moved["продажи_лаг1"]).to_numpy()
    base = moved["продажи_лаг1"].to_numpy() >= 1
    md, fd = md[base], fd[base]
    sig = np.abs(md) > 0.5
    sign_pct = float((np.sign(md[sig]) == np.sign(fd[sig])).mean() * 100)

    return {"seed": seed,
            "WMAPE_проц": round(wmape_pct, 2),
            "эффект_оптимизации_проц": round(effect_pct, 2),
            "совпадение_знака_реакции_проц": round(sign_pct, 1),
            "важность_признаков": importance}


def main() -> None:
    w = pd.read_pickle(ROOT / "data" / "weekly_sales.pkl")
    test = pd.read_pickle(ROOT / "data" / "test_predictions.pkl") \
        .reset_index(drop=True)

    runs = [run(seed, w, test) for seed in range(N_RUNS)]

    def rng(key):
        vals = [r[key] for r in runs]
        return [min(vals), max(vals)]

    out = {
        "описание": "плацебо-модели обучены на ценах, перемешанных по "
                    "времени внутри товара ТОЛЬКО в обучающем периоде "
                    "(исходная согласованная во времени связь цены и "
                    "продаж разрушена, случайные корреляции возможны; "
                    "временная утечка исключена); все оценки — на "
                    "реальном тесте; диагностический отрицательный "
                    "контроль, не статистический перестановочный тест",
        "прогонов": N_RUNS,
        "прогоны": runs,
        "диапазоны": {
            "WMAPE_проц": rng("WMAPE_проц"),
            "эффект_оптимизации_проц": rng("эффект_оптимизации_проц"),
            "совпадение_знака_реакции_проц":
                rng("совпадение_знака_реакции_проц"),
        },
        "для_сравнения_реальная_модель": {
            "WMAPE_проц": json.loads((REPORTS / "метрики_модели.json")
                                     .read_text(encoding="utf-8"))[
                "модель"]["WMAPE_проц"],
            "эффект_оптимизации_проц": json.loads(
                (REPORTS / "экономический_эффект.json")
                .read_text(encoding="utf-8"))[
                "эффект_к_фактической_цене_проц"],
            "совпадение_знака_реакции_проц": json.loads(
                (REPORTS / "проверка_ценовой_чувствительности.json")
                .read_text(encoding="utf-8"))[
                "совпадение_знака_реакции_проц"],
        },
    }
    (REPORTS / "плацебо_тест.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
