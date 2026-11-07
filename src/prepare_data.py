# -*- coding: utf-8 -*-
"""
Модуль подготовки данных для системы динамического ценообразования.

Вход:  исходный датасет продаж интернет-магазина (xlsx), строка = продажа
       (товар, день, категория, акция, количество, сумма закупки, сумма продажи).
Выход: агрегированный набор «товар-неделя» (pickle) + отчёт о качестве данных.

Этапы:
 1. Загрузка и кэширование xlsx -> pickle (ускорение повторных запусков).
 2. Очистка: удаление строк с нулевыми/отрицательными ценами и количеством.
 3. Агрегация дублей «товар+день» (несколько продаж в день).
 4. Агрегация к неделе: сумма количества, средневзвешенные цены, доля акций.
 5. Фильтрация товаров с короткой историей продаж.
 6. Сохранение результата и отчёта.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Пути ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # папка Прототип
RAW_XLSX = ROOT.parent / "Датасет.xlsx"             # исходник лежит в корне проекта
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

MIN_WEEKS_WITH_SALES = 20   # товар должен иметь продажи минимум в 20 неделях
MAX_QTY_PER_ROW = 100       # отсечение оптовых выбросов (медиана = 2 шт.)


def load_raw() -> pd.DataFrame:
    """Загружает исходный xlsx с кэшированием в pickle."""
    cache = DATA_DIR / "raw_cache.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_excel(RAW_XLSX)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Очистка данных; возвращает очищенный датафрейм и статистику удалений."""
    stats = {"строк_исходно": len(df)}
    df = df.copy()
    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
    stats["неудалось_распарсить_дату"] = int(df["Дата"].isna().sum())
    df = df.dropna(subset=["Дата"])

    df = df[df["Количество проданных товаров"] > 0]
    stats["удалено_нулевое_количество"] = stats["строк_исходно"] - len(df)

    # цены за единицу
    df["цена_продажи"] = df["Сумма продажи"] / df["Количество проданных товаров"]
    df["цена_закупки"] = df["Сумма закупки"] / df["Количество проданных товаров"]
    bad_price = (df["цена_продажи"] <= 0) | (df["цена_закупки"] <= 0)
    stats["удалено_нулевые_цены"] = int(bad_price.sum())
    df = df[~bad_price]

    # оптовые выбросы: строки с аномально большим количеством
    wholesale = df["Количество проданных товаров"] > MAX_QTY_PER_ROW
    stats["удалено_оптовых_выбросов"] = int(wholesale.sum())
    df = df[~wholesale]

    # строки с отрицательной маржой не удаляем (это реальные распродажи),
    # но считаем их для отчёта
    stats["строк_с_отрицательной_маржой_осталось"] = int(
        (df["цена_продажи"] < df["цена_закупки"]).sum()
    )
    stats["строк_после_очистки"] = len(df)
    return df, stats


def aggregate_day(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Агрегация нескольких продаж одного товара в один день."""
    before = len(df)
    g = df.groupby(["Наименование", "Дата"], as_index=False).apply(
        lambda x: pd.Series({
            "Категория": x["Категория"].iloc[0],
            "количество": x["Количество проданных товаров"].sum(),
            "цена_продажи": np.average(x["цена_продажи"],
                                       weights=x["Количество проданных товаров"]),
            "цена_закупки": np.average(x["цена_закупки"],
                                       weights=x["Количество проданных товаров"]),
            "доля_акций": x["Акция"].mean(),
        }), include_groups=False)
    return g, before - len(g)


def aggregate_week(day: pd.DataFrame) -> pd.DataFrame:
    """Агрегация к уровню «товар-неделя» (ISO-неделя)."""
    day = day.copy()
    iso = day["Дата"].dt.isocalendar()
    day["год"] = iso["year"].astype(int)
    day["неделя"] = iso["week"].astype(int)
    w = day.groupby(["Наименование", "год", "неделя"], as_index=False).apply(
        lambda x: pd.Series({
            "Категория": x["Категория"].iloc[0],
            "количество": x["количество"].sum(),
            "цена_продажи": np.average(x["цена_продажи"], weights=x["количество"]),
            "цена_закупки": np.average(x["цена_закупки"], weights=x["количество"]),
            "доля_акций": x["доля_акций"].mean(),
            "дней_с_продажами": len(x),
        }), include_groups=False)
    # понедельник недели — удобная метка времени
    w["дата_недели"] = pd.to_datetime(
        w["год"].astype(str) + w["неделя"].astype(str) + "1", format="%G%V%u")
    return w.sort_values(["Наименование", "дата_недели"]).reset_index(drop=True)


def filter_products(w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Оставляем товары с достаточной историей продаж."""
    weeks = w.groupby("Наименование")["неделя"].count()
    keep = weeks[weeks >= MIN_WEEKS_WITH_SALES].index
    stats = {
        "товаров_всего": int(w["Наименование"].nunique()),
        "товаров_после_фильтра": int(len(keep)),
        "мин_недель_с_продажами": MIN_WEEKS_WITH_SALES,
    }
    return w[w["Наименование"].isin(keep)].reset_index(drop=True), stats


def main() -> None:
    report = {}

    raw = load_raw()
    clean_df, clean_stats = clean(raw)
    report["очистка"] = clean_stats

    day_df, n_dup = aggregate_day(clean_df)
    report["агрегация_дня"] = {"объединено_дублей": int(n_dup),
                               "строк_день": len(day_df)}

    week_df = aggregate_week(day_df)
    week_df, filt_stats = filter_products(week_df)
    report["фильтрация_товаров"] = filt_stats

    # итоговая статистика
    cv = (week_df.groupby("Наименование")["цена_продажи"]
          .agg(["std", "mean"]))
    report["итог"] = {
        "строк_товар_неделя": len(week_df),
        "период": f'{week_df["дата_недели"].min():%d.%m.%Y} — '
                  f'{week_df["дата_недели"].max():%d.%m.%Y}',
        "недель_всего": int(week_df["дата_недели"].nunique()),
        "категорий": int(week_df["Категория"].nunique()),
        "товаров_с_CV_цены_более_10проц": int((cv["std"] / cv["mean"] > 0.10).sum()),
        "медиана_маржи_проц": round(float(
            ((week_df["цена_продажи"] - week_df["цена_закупки"])
             / week_df["цена_продажи"] * 100).median()), 2),
    }

    out = DATA_DIR / "weekly_sales.pkl"
    week_df.to_pickle(out)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "подготовка_данных.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {out} ({out.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    sys.exit(main())
