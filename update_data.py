"""
update_data.py
==============
Descarga precios históricos de Yahoo Finance y los guarda como CSVs en data/.
Lo ejecuta GitHub Actions cada semana automáticamente.
También puedes ejecutarlo tú manualmente: python update_data.py
"""

import os
import time
import datetime
import requests
import pandas as pd

# ============================================================
# TICKERS A DESCARGAR
# Añade aquí todos los tickers que uses en la app
# ============================================================
TICKERS = [
    # Escenario A
    "SPY",
    "GLD",
    # Escenario B
    "IBCF.DE",
    # Benchmark
    "EURUSD=X",
    # Añade más aquí si los usas en la app:
    # "EUNL.DE", "QQQ", "BTC-USD", ...
]

START_DATE = "2005-01-01"
DATA_DIR   = "data"


def get_crumb_and_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com",
    })
    try:
        session.get("https://finance.yahoo.com", timeout=15)
        time.sleep(1)
    except Exception:
        pass
    crumb = None
    try:
        r = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
        if r.status_code == 200 and r.text and "<" not in r.text:
            crumb = r.text.strip()
            print(f"  Crumb obtenido: {crumb[:10]}...")
    except Exception:
        pass
    return session, crumb


def download_ticker(ticker: str, start: str, session, crumb) -> pd.Series:
    ts1 = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp())
    ts2 = int(datetime.datetime.now().timestamp())
    params = {"period1": ts1, "period2": ts2, "interval": "1d"}
    if crumb:
        params["crumb"] = crumb

    for base in ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]:
        try:
            r = session.get(f"{base}/v8/finance/chart/{ticker}", params=params, timeout=20)
            r.raise_for_status()
            data   = r.json()
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            try:
                closes = result["indicators"]["adjclose"][0]["adjclose"]
            except Exception:
                closes = result["indicators"]["quote"][0]["close"]
            dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
            s = pd.Series(closes, index=dates, name=ticker).astype(float).dropna()
            if len(s) > 10:
                return s
        except Exception as e:
            print(f"    Fallo en {base}: {e}")
            time.sleep(1)

    raise ValueError(f"No se pudieron descargar datos para {ticker}")


def update_ticker_csv(ticker: str, session, crumb):
    safe_name = ticker.replace("=", "_").replace("/", "_")
    path = os.path.join(DATA_DIR, f"{safe_name}.csv")

    # Cargar existente si hay
    existing = None
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            existing.index = pd.to_datetime(existing.index)
            print(f"  {ticker}: CSV existente con {len(existing)} filas, última fecha: {existing.index.max().date()}")
        except Exception:
            existing = None

    # Descargar desde el principio (siempre, para tener histórico completo)
    print(f"  {ticker}: descargando desde {START_DATE}...")
    new_data = download_ticker(ticker, START_DATE, session, crumb)

    # Combinar con existente (por si hay datos más recientes)
    if existing is not None:
        combined = pd.concat([existing.iloc[:, 0] if isinstance(existing, pd.DataFrame) else existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_data

    combined.name = ticker
    combined.to_csv(path, header=[ticker])
    print(f"  {ticker}: ✅ guardado en {path} ({len(combined)} filas, hasta {combined.index.max().date()})")
    return combined


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Actualizando datos — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    session, crumb = get_crumb_and_session()

    ok, fail = [], []
    for ticker in TICKERS:
        print(f"→ {ticker}")
        try:
            update_ticker_csv(ticker, session, crumb)
            ok.append(ticker)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail.append(ticker)
        time.sleep(0.8)  # pausa entre peticiones

    print(f"\n{'='*50}")
    print(f"✅ OK ({len(ok)}): {ok}")
    if fail:
        print(f"❌ Fallidos ({len(fail)}): {fail}")
    print(f"{'='*50}\n")

    # Escribir manifiesto con fecha de última actualización
    manifest_path = os.path.join(DATA_DIR, "last_update.txt")
    with open(manifest_path, "w") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
    print(f"Manifiesto escrito en {manifest_path}")

    if fail:
        exit(1)


if __name__ == "__main__":
    main()
