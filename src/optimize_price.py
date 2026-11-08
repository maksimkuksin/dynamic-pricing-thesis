# -*- coding: utf-8 -*-
"""
Модуль оптимизации цены и оценки экономического эффекта.

Вход:  модель спроса (models/demand_lgbm.txt + demand_lgbm_features.json),
       прогнозы тестового периода (data/test_predictions.pkl),
       недельные данные (data/weekly_sales.pkl).
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
import sys
from pathlib import Path

import lightgbm as lgb
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
    # model_str вместо model_file: LightGBM C API не принимает не-ASCII пути
    model = lgb.Booster(model_str=(ROOT / "models" / "demand_lgbm.txt")
                        .read_text(encoding="utf-8"))
    features = json.loads(
        (ROOT / "models" / "demand_lgbm_features.json")
        .read_text(encoding="utf-8"))
    test = pd.read_pickle(DATA / "test_predictions.pkl")
    return model, features, test


def set_price(df: pd.DataFrame, price: np.ndarray) -> pd.DataFrame:
    """Пересчёт ценозависимых признаков под заданную цену."""
    out = df.copy()
    out["цена_продажи"] = price
    # среднее категории — скользящее за 4 прошлые недели (без утечки),
    # сохранено в столбце кат_сред_4н при формировании признаков
    out["цена_к_средней_категории"] = price / out["кат_сред_4н"]
    out["маржа_проц"] = (price - out["цена_закупки"]) / price * 100
    return out


def expected_profit(model, features, base: pd.DataFrame,
                    price: np.ndarray) -> np.ndarray:
    """Ожидаемая прибыль стратегии = (цена − себестоимость) × прогноз спроса."""
    rows = set_price(base, price)
    q = np.expm1(model.predict(rows[features])).clip(min=0)
    return (price - base["цена_закупки"].to_numpy()) * q, q


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _pct(x: float, digits: int = 1) -> str:
    """Проценты с десятичной запятой и типографским минусом."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def build_report_md(s: dict) -> str:
    """Текстовый отчёт об экономическом эффекте из сводки json."""
    p = s["ожидаемая_прибыль"]
    r = s["рекомендации"]
    e = s["оценка_эластичности"]
    under = (1 - p["фактическая_цена"]
             / s["фактически_реализованная_прибыль_за_период"]) * 100
    lines = [
        "# Отчёт об оптимизации цен и оценке экономического эффекта "
        "(модуль optimize_price.py)",
        "",
        f"Проверочный период — последние 8 недель данных, "
        f"{_fmt_money(s['строк_товар_неделя'])} позиций «товар-неделя». "
        "Отчёт сгенерирован автоматически из reports/экономический_эффект.json.",
        "",
        "## Метод",
        "Для каждой позиции перебирается сетка цен ±15 % от фактической "
        "(шаг 1 %). Для каждой цены модель спроса прогнозирует количество; "
        "считается ожидаемая прибыль (цена − себестоимость) × спрос. "
        "Рекомендуется цена с максимумом прибыли при ограничениях: "
        "маржа ≥ 10 %, изменение цены ≤ 15 %.",
        "",
        "## Результаты",
        "| Стратегия | Ожидаемая прибыль за 8 недель, руб. |",
        "|---|---|",
        f"| Фактическая цена (оценка модели) | {_fmt_money(p['фактическая_цена'])} |",
        f"| «Себестоимость + 30 %» | {_fmt_money(p['себестоимость_плюс_30проц'])} |",
        f"| **Рекомендованная цена** | **{_fmt_money(p['рекомендованная_цена'])}** |",
        "",
        f"- Эффект к фактической цене: "
        f"**+{_pct(s['эффект_к_фактической_цене_проц'])} %** "
        "(по оценке модели).",
        f"- Эффект к стратегии «себестоимость + 30 %»: "
        f"**+{_pct(s['эффект_к_cost_plus_проц'])} %**.",
        f"- Фактически реализованная прибыль за период (по данным продаж): "
        f"{_fmt_money(s['фактически_реализованная_прибыль_за_период'])} руб.",
        f"- Рекомендации: изменение цены — "
        f"{_pct(r['доля_изменений_цены_проц'])} % позиций (повышение "
        f"{_pct(r['доля_повышений_проц'])} %, понижение "
        f"{_pct(r['доля_понижений_проц'])} %); медианное изменение "
        f"+{r['медиана_изменения_проц']:.0f} %.",
        f"- Оценённая эластичность спроса: медиана **{_pct(e['медиана'], 2)}**; "
        f"неэластичных позиций (|E| < 1) — "
        f"{_pct(e['доля_неэластичных_позиций_проц'])} %.",
        "",
        "## Интерпретация и ограничения (обязательно отразить в ВКР)",
        "1. **Эффект — модельная оценка, а не факт внедрения.** Спрос при "
        "нереализованных ценах оценивается моделью на наблюдаемых данных; "
        "для подтверждения нужен A/B-тест. Цена выбирается по максимуму "
        "ожидаемой прибыли и оценивается той же моделью, поэтому оценка "
        "смещена вверх («проклятие победителя»); устойчивость вывода "
        "проверена в reports/анализ_чувствительности.json.",
        f"2. **Модель занижает уровень спроса**: её оценка прибыли при "
        f"фактических ценах ниже фактически реализованной примерно на "
        f"{under:.0f} % (одна из возможных причин — обратное преобразование "
        f"логарифмированной цели). Поэтому стратегии "
        f"сравниваются между собой на одной модельной шкале, а не с фактом "
        f"напрямую.",
        f"3. Медианная эластичность {_pct(e['медиана'], 1)} находится в "
        "нижней части диапазона эмпирических оценок мета-анализа Tellis "
        "(среднее около −1,76 при значительном разбросе).",
        f"4. {s['строк_без_изменения_из_за_ограничений']} позиций из-за "
        "ограничения минимальной маржи остались без изменения цены.",
        "5. **Плацебо-тест** (placebo_test.py, reports/плацебо_тест.json): "
        "модели, обученные на перемешанных по времени ценах, показывают "
        "сопоставимый или больший модельный «эффект», поэтому величина "
        "эффекта характеризует границу ценовой сетки при слабом отклике "
        "спроса, а не доказанный результат внедрения.",
        "",
        "## Файлы",
        f"- Рекомендации: reports/рекомендации_цен.csv "
        f"({_fmt_money(s['строк_товар_неделя'])} строк: товар, неделя, "
        "фактическая и рекомендуемая цена, прогноз спроса, ожидаемая прибыль).",
        "- Сводка: reports/экономический_эффект.json.",
        "- Устойчивость эффекта: reports/анализ_чувствительности.json "
        "(модуль sensitivity_analysis.py).",
        "",
    ]
    return "\n".join(lines)


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
            # доли позиций, чей оптимум попал точно на границу сетки ±15 %
            "доля_на_верхней_границе_сетки_проц": round(float(
                (change >= MAX_CHANGE * 100 - 1e-6).mean() * 100), 1),
            "доля_на_нижней_границе_сетки_проц": round(float(
                (change <= -MAX_CHANGE * 100 + 1e-6).mean() * 100), 1),
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
    # markdown-отчёт генерируется из той же сводки, что и json, —
    # текстовый отчёт не может разойтись с машиночитаемыми числами
    (REPORTS / "экономический_эффект.md").write_text(
        build_report_md(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nПримеры рекомендаций (топ-5 по прибыли):")
    print(rec.nlargest(5, "прибыль_ожид").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
