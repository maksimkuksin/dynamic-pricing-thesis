# -*- coding: utf-8 -*-
"""Пересъёмка скриншотов интерфейса Streamlit (рисунки 9–12 ВКР).

Временно поднимает `streamlit run src/app.py`, делает скриншоты
Playwright'ом и завершает сервер. Запуск из корня проекта «Прототип».
"""
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "Глава3" / "рисунки"
PORT = 8599
URL = f"http://localhost:{PORT}"

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "src/app.py",
     "--server.port", str(PORT), "--server.headless", "true"],
    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    # ждём поднятия сервера
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("streamlit не поднялся")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.goto(URL)
        page.wait_for_selector("text=Позиций с рекомендациями",
                               timeout=120000)
        page.wait_for_timeout(4000)

        # рис. 9 — сводка + вкладка «Рекомендации»
        page.screenshot(path=str(OUT / "интерфейс_рекомендации.png"))

        # вкладка «Карточка товара»
        page.click("text=Карточка товара")
        page.wait_for_selector("text=История: цена и продажи по неделям",
                               timeout=60000)
        page.wait_for_timeout(4000)
        # рис. 10 — карточка: метрики и история
        page.screenshot(path=str(OUT / "интерфейс_карточка.png"))

        # зафиксируем два решения для журнала
        page.click("text=Принять рекомендацию")
        page.wait_for_timeout(1500)
        page.click("text=Отклонить рекомендацию")
        page.wait_for_timeout(1500)

        # рис. 11 — кривые «цена → спрос → прибыль»
        page.set_viewport_size({"width": 1500, "height": 1100})
        curves = page.locator("text=Кривая «цена → спрос → прибыль»")
        curves.scroll_into_view_if_needed()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "интерфейс_кривые.png"))

        # рис. 12 — журнал решений
        page.set_viewport_size({"width": 1500, "height": 950})
        page.click("text=Журнал решений")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "интерфейс_журнал.png"))

        browser.close()
    print("скриншоты сохранены")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
