# -*- coding: utf-8 -*-
"""
Веб-интерфейс менеджера по ценообразованию (Streamlit).

Запуск:  streamlit run Прототип/src/app.py
         (рабочая папка — корень проекта)

Функции:
 - сводка: число рекомендаций, ожидаемый эффект, доля изменений;
 - таблица рекомендаций с фильтрами (категория, тип изменения, неделя);
 - карточка товара: история цены и продаж, кривая «цена -> спрос -> прибыль»;
 - журнал решений: менеджер принимает или отклоняет рекомендацию
   (сохранение в SQLite, Прототип/data/журнал_решений.db).
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DB = DATA / "журнал_решений.db"


# --- загрузка -------------------------------------------------------------
@st.cache_resource
def load_model():
    model = lgb.Booster(model_str=(ROOT / "models" / "demand_lgbm.txt")
                        .read_text(encoding="utf-8"))
    features = json.loads(
        (ROOT / "models" / "demand_lgbm_features.json")
        .read_text(encoding="utf-8"))
    return {"model": model, "features": features}


@st.cache_data
def load_data():
    rec = pd.read_csv(REPORTS / "рекомендации_цен.csv",
                      parse_dates=["дата_недели"])
    test = pd.read_pickle(DATA / "test_predictions.pkl")
    weekly = pd.read_pickle(DATA / "weekly_sales.pkl")
    return rec, test, weekly


def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS решения (
            время TEXT, товар TEXT, неделя TEXT,
            цена_была REAL, цена_новая REAL, решение TEXT)""")
        con.commit()


def log_decision(row, decision: str):
    with sqlite3.connect(DB) as con:
        con.execute(
            "INSERT INTO решения VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"),
             row["Наименование"], str(row["дата_недели"].date()),
             float(row["цена_факт"]), float(row["цена_реком"]), decision))
        con.commit()


def load_log() -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql("SELECT * FROM решения ORDER BY время DESC", con)


# --- кривая «цена -> спрос -> прибыль» ------------------------------------
def price_curve(bundle, base: pd.DataFrame) -> pd.DataFrame:
    """base — однострочный DataFrame с признаками позиции (dtype сохранены)."""
    model, features = bundle["model"], bundle["features"]
    row = base.iloc[0]
    grid = np.arange(0.85, 1.151, 0.01)
    prices = row["цена_продажи"] * grid
    expanded = base.loc[np.repeat(base.index, len(grid))].reset_index(drop=True)
    cat_mean = row["цена_продажи"] / row["цена_к_средней_категории"]
    expanded["цена_продажи"] = prices
    expanded["цена_к_средней_категории"] = prices / cat_mean
    expanded["маржа_проц"] = ((prices - row["цена_закупки"]) / prices) * 100
    q = np.expm1(model.predict(expanded[features])).clip(min=0)
    profit = (prices - row["цена_закупки"]) * q
    return pd.DataFrame({"цена": prices.round(2), "спрос": q.round(1),
                         "прибыль": profit.round(0)})


# --- интерфейс --------------------------------------------------------------
st.set_page_config(page_title="Динамическое ценообразование", layout="wide")
st.title("Интеллектуальная система динамического ценообразования")
st.caption("Прототип: прогноз спроса (LightGBM) + оптимизация цены "
           "по ожидаемой маржинальной прибыли. Проверочный период — "
           "последние 8 недель данных.")

bundle = load_model()
rec, test, weekly = load_data()
init_db()

total_profit = rec["прибыль_ожид"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Позиций с рекомендациями", f"{len(rec):,}".replace(",", " "))
c2.metric("Ожидаемая прибыль за период", f"{total_profit / 1e6:,.1f} млн ₽"
          .replace(",", " "))
c3.metric("Рекомендовано повышений",
          f"{(rec['изменение_проц'] > 0.5).mean() * 100:.0f}%")
c4.metric("Рекомендовано понижений",
          f"{(rec['изменение_проц'] < -0.5).mean() * 100:.0f}%")

tab_rec, tab_card, tab_log = st.tabs(
    ["Рекомендации", "Карточка товара", "Журнал решений"])

with tab_rec:
    f1, f2, f3 = st.columns(3)
    cats = ["Все"] + sorted(rec["Категория"].unique().tolist())
    cat = f1.selectbox("Категория", cats)
    kind = f2.selectbox("Тип изменения",
                        ["Все", "Повышение", "Понижение", "Без изменения"])
    weeks = sorted(rec["дата_недели"].dt.date.unique())
    week = f3.selectbox("Неделя", ["Все"] + [str(w) for w in weeks])

    view = rec.copy()
    if cat != "Все":
        view = view[view["Категория"] == cat]
    if kind == "Повышение":
        view = view[view["изменение_проц"] > 0.5]
    elif kind == "Понижение":
        view = view[view["изменение_проц"] < -0.5]
    elif kind == "Без изменения":
        view = view[view["изменение_проц"].abs() <= 0.5]
    if week != "Все":
        view = view[view["дата_недели"].dt.date.astype(str) == week]

    st.dataframe(
        view[["Наименование", "Категория", "дата_недели", "цена_факт",
              "цена_реком", "изменение_проц", "спрос_прогноз",
              "прибыль_ожид"]].rename(columns={
                  "Наименование": "Товар", "дата_недели": "Неделя",
                  "цена_факт": "Цена сейчас, ₽",
                  "цена_реком": "Рекомендуемая цена, ₽",
                  "изменение_проц": "Изменение, %",
                  "спрос_прогноз": "Прогноз спроса, шт./нед.",
                  "прибыль_ожид": "Ожидаемая прибыль, ₽"}),
        width="stretch", height=420)
    st.caption(f"Показано строк: {len(view)}")

with tab_card:
    names = sorted(rec["Наименование"].unique())
    name = st.selectbox("Товар", names)
    wk = st.selectbox(
        "Неделя рекомендации",
        sorted(rec.loc[rec["Наименование"] == name, "дата_недели"]
               .dt.date.unique()))
    row_rec = rec[(rec["Наименование"] == name)
                  & (rec["дата_недели"].dt.date == wk)].iloc[0]
    row_feat = test[(test["Наименование"] == name)
                    & (test["дата_недели"].dt.date == wk)].head(1)

    a, b, c = st.columns(3)
    a.metric("Цена сейчас", f"{row_rec['цена_факт']:,.0f} ₽".replace(",", " "))
    b.metric("Рекомендуемая цена",
             f"{row_rec['цена_реком']:,.0f} ₽".replace(",", " "),
             f"{row_rec['изменение_проц']:+.1f}%")
    c.metric("Ожидаемая прибыль",
             f"{row_rec['прибыль_ожид']:,.0f} ₽".replace(",", " "))

    hist = weekly[weekly["Наименование"] == name]
    st.subheader("История: цена и продажи по неделям")
    st.line_chart(hist.set_index("дата_недели")[["цена_продажи"]],
                  y_label="Цена, ₽")
    st.line_chart(hist.set_index("дата_недели")[["количество"]],
                  y_label="Продажи, шт.")

    st.subheader("Кривая «цена → спрос → прибыль» (по модели)")
    curve = price_curve(bundle, row_feat)
    st.line_chart(curve.set_index("цена")[["спрос"]],
                  y_label="Прогноз спроса, шт.")
    st.line_chart(curve.set_index("цена")[["прибыль"]],
                  y_label="Ожидаемая прибыль, ₽")

    d1, d2 = st.columns(2)
    if d1.button("Принять рекомендацию", type="primary"):
        log_decision(row_rec, "принято")
        st.success("Решение записано в журнал.")
    if d2.button("Отклонить рекомендацию"):
        log_decision(row_rec, "отклонено")
        st.info("Решение записано в журнал.")

with tab_log:
    log = load_log()
    st.dataframe(log, width="stretch")
    st.caption(f"Всего решений: {len(log)}. Хранение: SQLite, "
               "Прототип/data/журнал_решений.db")
