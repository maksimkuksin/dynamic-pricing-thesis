# -*- coding: utf-8 -*-
"""Проверка ценовой чувствительности модели на реальных изменениях цен.

Идея: в проверочном периоде находим позиции «товар-неделя», где цена
фактически изменилась к прошлой неделе. Для каждой такой позиции сравниваем:
  - фактическое изменение продаж (к неделе назад);
  - изменение продаж, которое предсказывает модель при фактической цене
    относительно прогноза при прежней цене (прогноз без изменения цены).
Совпадение знака и корреляция величин — описательная проверка того,
что реакция модели на цену согласуется с реакцией спроса в данных.
Проверка не является причинной оценкой: на фактические продажи влияют
и прочие факторы (сезонность, запасы, промо), что учитывается
при интерпретации результатов в тексте работы.
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

MIN_PRICE_CHANGE_PCT = 2.0   # порог существенного изменения цены, %


def main() -> None:
    model = lgb.Booster(model_str=(ROOT / "models" / "demand_lgbm.txt")
                        .read_text(encoding="utf-8"))
    features = json.loads(
        (ROOT / "models" / "demand_lgbm_features.json")
        .read_text(encoding="utf-8"))
    test = pd.read_pickle(DATA / "test_predictions.pkl")

    moved = test[test["изменение_цены_проц"].abs() >= MIN_PRICE_CHANGE_PCT].copy()

    # прогноз при ПРЕЖНЕЙ цене (цена_лаг1)
    cf = moved.copy()
    cf["цена_продажи"] = cf["цена_лаг1"]
    cf["изменение_цены_проц"] = 0.0
    cf["цена_к_средней_категории"] = cf["цена_лаг1"] / cf["кат_сред_4н"]
    cf["маржа_проц"] = ((cf["цена_лаг1"] - cf["цена_закупки"])
                        / cf["цена_лаг1"] * 100)
    q_old = np.expm1(model.predict(cf[features])).clip(min=0)

    q_new = moved["прогноз"].to_numpy()          # прогноз при фактической цене
    model_delta = q_new - q_old                   # реакция модели на изменение цены
    fact_delta = (moved["количество"]
                  - moved["продажи_лаг1"]).to_numpy()  # фактическое изменение

    # устойчивая подвыборка: недели с ненулевой базой продаж
    base = moved["продажи_лаг1"].to_numpy() >= 1
    moved_b = moved[base]
    md, fd = model_delta[base], fact_delta[base]
    q_old_b = q_old[base]

    # знаковое согласие (для существенной реакции модели)
    sig = np.abs(md) > 0.5
    n_valid_sign = int(sig.sum())
    sign_agree = float((np.sign(md[sig]) == np.sign(fd[sig])).mean() * 100)

    # отдельно по направлению изменения цены
    up = moved_b["изменение_цены_проц"].to_numpy() > 0
    def agree(mask):
        m = mask & sig
        if m.sum() == 0:
            return None
        return round(float((np.sign(md[m]) == np.sign(fd[m])).mean() * 100), 1)

    # ранговая корреляция величин изменений (Spearman, без scipy)
    r_model = pd.Series(md).rank().to_numpy()
    r_fact = pd.Series(fd).rank().to_numpy()
    spearman = float(np.corrcoef(r_model, r_fact)[0, 1])

    res = {
        "порог_изменения_цены_проц": MIN_PRICE_CHANGE_PCT,
        "позиций_с_изменением_цены": int(len(moved)),
        "доля_от_проверочного_набора_проц": round(len(moved) / len(test) * 100, 1),
        "позиций_с_ненулевой_базой": int(base.sum()),
        "позиций_в_расчёте_совпадения_знака": n_valid_sign,
        "совпадение_знака_реакции_проц": round(sign_agree, 1),
        "совпадение_знака_при_повышении_цены_проц": agree(up),
        "совпадение_знака_при_понижении_цены_проц": agree(~up),
        "ранговая_корреляция_Spearman": round(spearman, 3),
    }
    (REPORTS / "проверка_ценовой_чувствительности.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
