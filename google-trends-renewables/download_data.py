"""Descarga los datos de Google Trends (pytrends) y los guarda como CSV en data/.

Uso:  python download_data.py
Los CSV se cachean: el notebook funciona aunque Google limite las peticiones (HTTP 429).
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

KEYWORDS = ["solar energy", "wind energy", "hydropower"]
END = date.today()
START = END - timedelta(days=365 * 10)
TIMEFRAME = f"{START:%Y-%m-%d} {END:%Y-%m-%d}"
DATA_DIR = Path(__file__).parent / "data"


def with_retries(fn, tries: int = 6, wait: int = 20):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # pytrends lanza TooManyRequestsError / ResponseError
            print(f"  intento {i}/{tries} fallido: {exc}")
            time.sleep(wait * i)
    raise RuntimeError("Google Trends no respondió tras varios intentos")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    # retries=0/backoff_factor=0: pytrends usa 'method_whitelist', incompatible con urllib3>=2;
    # los reintentos los gestiona with_retries().
    pt = TrendReq(hl="en-US", tz=0, retries=0, backoff_factor=0)
    with_retries(lambda: pt.build_payload(KEYWORDS, timeframe=TIMEFRAME, geo=""))

    print("Interés en el tiempo (mundial)...")
    over_time = with_retries(pt.interest_over_time)
    over_time.to_csv(DATA_DIR / "interest_over_time.csv")

    time.sleep(5)
    print("Interés por país...")
    by_region = with_retries(lambda: pt.interest_by_region(resolution="COUNTRY", inc_low_vol=True, inc_geo_code=True))
    by_region.to_csv(DATA_DIR / "interest_by_region.csv")

    print(f"Listo: {len(over_time)} filas en el tiempo, {len(by_region)} países. Periodo {TIMEFRAME}")


if __name__ == "__main__":
    main()
