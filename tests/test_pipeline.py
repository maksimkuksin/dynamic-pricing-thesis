# -*- coding: utf-8 -*-
"""Автоматические тесты конвейера прототипа.

Запуск из корня проекта:  python -m pytest Прототип/tests -v
Тесты работают на уже построенных артефактах (data/*.pkl, models/*,
reports/*) и не перезапускают обучение.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from train_model import FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def weekly():
    return pd.read_pickle(ROOT / "data" / "weekly_sales.pkl")


@pytest.fixture(scope="module")
def test_pred():
    return pd.read_pickle(ROOT / "data" / "test_predictions.pkl")


@pytest.fixture(scope="module")
def model_bundle():
    model = lgb.Booster(model_str=(ROOT / "models" / "demand_lgbm.txt")
                        .read_text(encoding="utf-8"))
    features = json.loads((ROOT / "models" / "demand_lgbm_features.json")
                          .read_text(encoding="utf-8"))
    return model, features


def test_no_nonpositive_prices(weekly):
    """В недельных данных нет нулевых и отрицательных цен."""
    assert (weekly["цена_продажи"] > 0).all()
    assert (weekly["цена_закупки"] > 0).all()


def test_no_negative_quantities(weekly):
    """Количество продаж неотрицательно."""
    assert (weekly["количество"] >= 0).all()


def test_test_set_has_all_features(test_pred):
    """Проверочный набор содержит все признаки модели без пропусков."""
    for f in FEATURES:
        assert f in test_pred.columns, f"нет столбца {f}"
        assert not test_pred[f].isna().any(), f"пропуски в {f}"


def test_predictions_nonnegative(test_pred):
    """Прогнозы спроса неотрицательны."""
    assert (test_pred["прогноз"] >= 0).all()


def test_recommendations_constraints(test_pred):
    """Рекомендации цен соблюдают ограничения оптимизатора."""
    rec = pd.read_csv(ROOT / "reports" / "рекомендации_цен.csv",
                      encoding="utf-8-sig")
    assert len(rec) == len(test_pred), "число строк не совпадает с тестом"
    # маржа рекомендованной цены не ниже 10 % (кроме строк, где все
    # кандидаты были запрещены и оставлена фактическая цена)
    margin = (rec["цена_реком"] - rec["себестоимость"]) / rec["цена_реком"]
    changed = rec["изменение_проц"].abs() > 0.5
    assert (margin[changed] >= 0.10 - 1e-6).all(), "маржа ниже 10 %"
    # изменение цены не выходит за ±15 % (с допуском на округление)
    assert (rec["изменение_проц"].abs() <= 15.05).all()


def test_monotone_price_response(model_bundle, test_pred):
    """Монотонность: рост относительной цены не повышает прогноз спроса.

    Для выборки позиций прогноз при цене +15 % не превышает прогноз
    при цене −15 % (с небольшим допуском на численные эффекты).
    """
    from optimize_price import set_price
    model, features = model_bundle
    sample = test_pred.sample(200, random_state=42).reset_index(drop=True)
    p0 = sample["цена_продажи"].to_numpy()
    q_down = model.predict(set_price(sample, p0 * 0.85)[features])
    q_up = model.predict(set_price(sample, p0 * 1.15)[features])
    assert (q_up <= q_down + 1e-6).mean() >= 0.99


def test_monotone_constraints_declared():
    """В модели заявлено монотонное ограничение −1 на ценовые признаки."""
    from train_model import PRICE_FEATURES, monotone_constraints
    mc = dict(zip(FEATURES, monotone_constraints()))
    for f in PRICE_FEATURES:
        assert mc[f] == -1
    assert "маржа_проц" not in FEATURES
    assert "цена_продажи" not in FEATURES


def test_metrics_files_exist():
    """Итоговые отчёты существуют и содержат ключевые поля."""
    metrics = json.loads((ROOT / "reports" / "метрики_модели.json")
                         .read_text(encoding="utf-8"))
    assert "модель" in metrics and "WMAPE_проц" in metrics["модель"]
    effect = json.loads((ROOT / "reports" / "экономический_эффект.json")
                        .read_text(encoding="utf-8"))
    assert "эффект_к_фактической_цене_проц" in effect
    assert np.isfinite(effect["эффект_к_фактической_цене_проц"])
