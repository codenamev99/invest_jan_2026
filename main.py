# main.py
import os
import time
import datetime
import logging
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")

# ==============================
# Config
# ==============================
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_ALPHA_VANTAGE_KEY_HERE")
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
# Free tier ~5 req/min; pace calls across threads.
AV_RATE_LIMIT_SECONDS = 13

_av_lock = threading.Lock()
_last_av_call_ts = [0.0]


# ==============================
# Helpers
# ==============================
def map_symbols_for_providers(symbol: str):
    """
    Return (yahoo_symbol, alpha_vantage_symbol).
    Yahoo uses BRK-B/BF-B style, while Alpha Vantage typically accepts BRK.B/BF.B.
    """
    yahoo_symbol = symbol.replace(".", "-")
    av_symbol = symbol  # keep dot for AV
    return yahoo_symbol, av_symbol


# ==============================
# Indicator helpers
# ==============================
def get_atr(hist, period=14):
    high_low = hist['High'] - hist['Low']
    high_close = (hist['High'] - hist['Close'].shift()).abs()
    low_close = (hist['Low'] - hist['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def get_rsi_wilder_local(hist, period=14, use_adj_close=False, exclude_last=False):
    """Local Wilder RSI fallback."""
    px_col = 'Adj Close' if use_adj_close and 'Adj Close' in hist.columns else 'Close'
    data = hist if not exclude_last else hist.iloc[:-1]

    delta = data[px_col].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def coerce_first_numeric(val):
    """
    Return the first numeric value from scalars, lists, tuples, ndarrays, or Series.
    Non-numeric or NaN returns None.
    """
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    elif isinstance(val, (np.ndarray, pd.Series)):
        val = val[0] if len(val) else None
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if not pd.isna(out) else None


def normalize_ohlc_columns(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten possible MultiIndex columns from yfinance (e.g., ('AAPL', 'Close')) to single-level
    OHLCV names so downstream selections return scalars instead of Series.
    """
    if isinstance(hist.columns, pd.MultiIndex):
        # Use the last level (typically OHLC names)
        hist = hist.copy()
        hist.columns = hist.columns.get_level_values(-1)
    return hist


def get_rsi_alpha_vantage(symbol, period=14, interval="daily", series_type="close"):
    """
    Fetch RSI from Alpha Vantage Technical Indicators API.
    """
    if not ALPHAVANTAGE_API_KEY or ALPHAVANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_KEY_HERE":
        raise RuntimeError("Alpha Vantage API key not set. Set ALPHAVANTAGE_API_KEY or edit the script.")

    params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": interval,        # "daily", "weekly", "monthly", or intraday like "60min"
        "time_period": period,       # default 14
        "series_type": series_type,  # "close", "open", "high", "low"
        "datatype": "json",
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    with _av_lock:
        now = time.time()
        wait_for = AV_RATE_LIMIT_SECONDS - (now - _last_av_call_ts[0])
        if wait_for > 0:
            time.sleep(wait_for)
        resp = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
        _last_av_call_ts[0] = time.time()

    if resp.status_code != 200:
        raise RuntimeError(f"Alpha Vantage HTTP {resp.status_code}")

    data = resp.json()
    if "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Note") or data.get("Information"))
    if "Technical Analysis: RSI" not in data:
        raise RuntimeError(data.get("Error Message") or "Unknown AV response; no RSI data.")

    ta = data["Technical Analysis: RSI"]
    if not ta:
        raise RuntimeError("Alpha Vantage returned empty RSI series.")

    latest_ts = max(ta.keys())
    rsi_str = ta[latest_ts].get("RSI")
    if rsi_str is None:
        raise RuntimeError("Alpha Vantage RSI field missing for latest bar.")
    return float(rsi_str)


# ==============================
# yfinance helpers
# ==============================
def fetch_yf_history(yahoo_symbol: str, period="6mo", interval="1d"):
    """
    Fetch history using yfinance default session; retry with download fallback.
    """
    for attempt in range(3):
        try:
            tk = yf.Ticker(yahoo_symbol)
            hist = tk.history(period=period, interval=interval, actions=False)
            if hist is not None and not hist.empty:
                return hist
            logging.debug(
                f"yfinance returned empty history for {yahoo_symbol} on attempt {attempt + 1} "
                f"(period={period}, interval={interval})."
            )
        except Exception as e:
            msg = str(e)
            if "possibly delisted" in msg:
                logging.debug(f"{yahoo_symbol} likely delisted or no data; skipping.")
                return None
            logging.debug(f"yfinance error for {yahoo_symbol} on attempt {attempt + 1}: {e}")

        # Fallback: try download API with a fresh session
        try:
            hist = yf.download(
                yahoo_symbol,
                period=period,
                interval=interval,
                progress=False,
            )
            if hist is not None and not hist.empty:
                logging.debug(f"yfinance download fallback succeeded for {yahoo_symbol} on attempt {attempt + 1}.")
                return hist
            logging.debug(f"yfinance download fallback empty for {yahoo_symbol} on attempt {attempt + 1}.")
        except Exception as dl_err:
            logging.debug(f"yfinance download fallback error for {yahoo_symbol} on attempt {attempt + 1}: {dl_err}")

        time.sleep(0.5)
    return None


# ==============================
# Ticker universe
# ==============================
def fetch_all_tickers():
    """
    Fetch NYSE-listed tickers from NASDAQ Trader's otherlisted feed with
    yahoo_fin fallback.
    """
    url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    logging.info("Fetching NYSE tickers from NASDAQ Trader feed...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep="|", dtype=str)
        df = df[df["ACT Symbol"].notna()]
        df = df[df["ACT Symbol"] != "File Creation Time"]
        df = df[df["Exchange"] == "N"]          # NYSE listings
        df = df[df["Test Issue"] == "N"]        # exclude test issues
        tickers = (
            df["ACT Symbol"]
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not tickers:
            raise ValueError("No tickers parsed from primary feed.")
        logging.info(f"Fetched {len(tickers)} NYSE tickers (primary source).")
        return tickers
    except Exception as primary_err:
        logging.warning(f"Primary NYSE ticker fetch failed: {primary_err}. Trying yahoo_fin fallback...")
        try:
            from yahoo_fin import stock_info as si
            raw_tickers = []
            if hasattr(si, "tickers_nyse"):
                raw_tickers = si.tickers_nyse()
            else:
                # tickers_nyse was removed in recent yahoo_fin; use available sets instead.
                fallback_funcs = ["tickers_other", "tickers_nasdaq", "tickers_sp500", "tickers_dow"]
                for fn in fallback_funcs:
                    if hasattr(si, fn):
                        try:
                            raw_tickers.extend(getattr(si, fn)())
                        except Exception as sub_err:
                            logging.warning(f"yahoo_fin {fn} fallback failed: {sub_err}")

            tickers = sorted({t.strip().upper() for t in raw_tickers if t})
            if not tickers:
                raise ValueError("No tickers returned from yahoo_fin fallbacks.")

            logging.info(f"Fetched {len(tickers)} tickers (yahoo_fin fallback set union).")
            return tickers
        except Exception as fallback_err:
            logging.error(f"Fallback NYSE ticker fetch failed: {fallback_err}")
            raise


def load_tickers_from_csv(csv_path="nyse_tickers.csv"):
    """
    Load tickers from a local CSV (first column or 'Ticker' column). One symbol per row.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ticker CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        raise ValueError(f"Ticker CSV {csv_path} is empty.")
    if "Ticker" in df.columns:
        series = df["Ticker"]
    else:
        series = df.iloc[:, 0]
    tickers = (
        series.astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    if not tickers:
        raise ValueError(f"No tickers parsed from CSV {csv_path}.")
    return tickers


# ==============================
# Analyze a single ticker
# ==============================
def analyze_ticker(symbol, ignore_thresholds=False, av_interval="daily", av_period=14, av_series_type="close"):
    try:
        yf_symbol, av_symbol = map_symbols_for_providers(symbol)

        hist = fetch_yf_history(yf_symbol, period="6mo", interval="1d")
        if hist is not None:
            hist = normalize_ohlc_columns(hist)
        tk = yf.Ticker(yf_symbol)

        if hist is None or len(hist) < 30:
            logging.debug(f"No usable history for {symbol}: hist=None or len={0 if hist is None else len(hist)}.")
            return None

        price_raw = hist['Close'].iloc[-1] if 'Close' in hist.columns else None
        price = coerce_first_numeric(price_raw)
        if price is None or price < 5:
            logging.debug(f"Price screen failed for {symbol}: price={price}.")
            return None

        # ATR %
        atr = get_atr(hist).iloc[-1]
        atr_pct = (atr / price) * 100 if pd.notna(atr) else None

        # RSI from Alpha Vantage (primary) with local fallback
        try:
            rsi = get_rsi_alpha_vantage(av_symbol, period=av_period, interval=av_interval, series_type=av_series_type)
        except Exception as av_err:
            logging.debug(f"Alpha Vantage RSI failed for {symbol}: {av_err}. Falling back to local Wilder RSI.")
            rsi = get_rsi_wilder_local(hist, period=av_period, use_adj_close=False, exclude_last=False).iloc[-1]

        # MACD (local)
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_series = ema12 - ema26
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_series.iloc[-1])
        signal_val = float(signal_series.iloc[-1])
        macd_gt_signal = macd_val > signal_val  # used for filtering, not for output

        # Beta (optional; may be None)
        try:
            raw_beta = tk.info.get('beta', None)
            beta = coerce_first_numeric(raw_beta)
        except Exception as info_err:
            logging.debug(f"yfinance info fetch failed for {symbol}: {info_err}")
            beta = None

        # Average Daily Volume
        avg_daily_vol = hist['Volume'].rolling(20).mean().iloc[-1]
        avg_daily_vol_val = coerce_first_numeric(avg_daily_vol)

        # Normalize scalar fields to avoid Series truth-value errors
        rsi_val = coerce_first_numeric(rsi)
        atr_pct_val = coerce_first_numeric(atr_pct)
        beta_val = coerce_first_numeric(beta)

        # Apply thresholds if not ignoring
        if not ignore_thresholds:
            if (not macd_gt_signal) \
               or (rsi_val is None or rsi_val < 50 or rsi_val > 70) \
               or (atr_pct_val is None or atr_pct_val <= 2) \
               or (beta_val is None or beta_val <= 1.2) \
               or (avg_daily_vol_val is None or avg_daily_vol_val <= 1_000_000):
                return None

        # === Output dict with requested column changes ===
        return {
            "Ticker": symbol,                                  # original symbol for display
            "Price": round(float(price), 2),
            "ATR%": round(float(atr_pct_val), 2) if atr_pct_val is not None else None,
            "RSI": round(float(rsi_val), 2) if rsi_val is not None else None,
            # Print MACD and Signal values (string keeps CSV friendly formatting)
            "MACD>Signal": f"{macd_val:.4f} / {signal_val:.4f}",
            # Print numeric 20-day avg volume instead of boolean
            "AvgDailyVol>1M": int(avg_daily_vol_val) if avg_daily_vol_val is not None else None,
            "Beta": round(float(beta_val), 2) if beta_val is not None else None,
        }

    except Exception:
        # Include stack trace to pinpoint any unexpected pandas truthiness or data issues
        logging.exception(f"Ticker {symbol} error")
    return None


# ==============================
# Run screener
# ==============================
def run_screener(tickers):
    results = []
    max_workers = 6
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(analyze_ticker, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            symbol = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logging.debug(f"Error for {symbol}: {e}")
            time.sleep(0.01)  # polite delay for yfinance (not AV)

    df = pd.DataFrame(results)

    if df.empty:
        logging.info("No candidates matched filters this run.")
        return df

    if "ATR%" in df.columns:
        df = df.sort_values(by="ATR%", ascending=False, na_position="last")

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"candidates_{ts}.csv"
    with open(out_path, 'w') as f:
        f.write(f"# Screener Parameters:\n")
        f.write(f"# MACD > Signal (values printed as 'MACD / Signal')\n")
        f.write(f"# RSI(14) from Alpha Vantage (daily, close)\n")
        f.write(f"# ATR% > 2\n")
        f.write(f"# Beta > 1.2\n")
        f.write(f"# AvgDailyVol > 1M (value printed)\n\n")
    df.to_csv(out_path, mode='a', index=False)
    logging.info(f"Found {len(df)} candidates. Saved to {out_path}")
    return df


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    print("Select mode:")
    print("1 - Scan all NYSE tickers (no ticker list CSV)")
    print("2 - Scan a specific ticker")
    print("3 - Generate NYSE ticker list CSV only")
    choice = input("Enter 1, 2 or 3: ").strip()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame()  # default empty for non-scan choices

    if choice == "1":
        tickers = load_tickers_from_csv("nyse_tickers.csv")
        logging.info(f"Loaded {len(tickers)} tickers from nyse_tickers.csv. Running screener...")
        df = run_screener(tickers)

    elif choice == "2":
        ticker_input = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
        res = analyze_ticker(ticker_input, ignore_thresholds=True)  # show all metrics
        if res:
            df = pd.DataFrame([res])
            out_path = f"candidates_{ticker_input}_{ts}.csv"
            with open(out_path, 'w') as f:
                f.write(f"# Screener Parameters:\n")
                f.write(f"# MACD > Signal (values printed as 'MACD / Signal')\n")
                f.write(f"# RSI(14) from Alpha Vantage (daily, close)\n")
                f.write(f"# ATR% > 2\n")
                f.write(f"# Beta > 1.2\n")
                f.write(f"# AvgDailyVol > 1M (value printed)\n\n")
            df.to_csv(out_path, mode='a', index=False)
            logging.info(f"Saved single ticker result to {out_path}")
        else:
            logging.info(f"Could not fetch metrics for {ticker_input}.")

    elif choice == "3":
        tickers = fetch_all_tickers()
        all_tickers_path = f"all_scanned_tickers_nyse_{ts}.csv"
        pd.DataFrame({"Ticker": tickers}).to_csv(all_tickers_path, index=False)
        logging.info(f"Saved all scanned tickers to {all_tickers_path}")

    else:
        print("Invalid choice. Exiting.")
        exit()

    if not df.empty:
        print(df.head(30))
