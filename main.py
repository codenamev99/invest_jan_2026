# main.py
import os
import time
import datetime
import logging
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def load_dotenv_simple(dotenv_path=".env"):
    """
    Minimal .env loader: KEY=VALUE per line, supports quotes and ignores comments.
    Existing environment variables are not overridden.
    """
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        logging.debug(f"Failed to load {dotenv_path}: {e}")


# Load local .env before reading config env vars
load_dotenv_simple()


log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")

# ==============================
# Config
# ==============================
SEC_BASE_URL = "https://data.sec.gov"
SEC_FILES_BASE_URL = "https://www.sec.gov"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Your Name your.email@example.com")
# SEC guidelines: keep requests well-paced and include contact info.
SEC_RATE_LIMIT_SECONDS = float(os.getenv("SEC_RATE_LIMIT_SECONDS", "0.2"))
SEC_TICKER_CACHE_TTL = int(os.getenv("SEC_TICKER_CACHE_TTL", "86400"))

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))

# Optional fast profile: prioritizes speed over retry friendliness.
FAST_PROFILE = os.getenv("FAST_PROFILE", "0") == "1"

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
SEC_PREFER_ANNUAL = os.getenv("SEC_PREFER_ANNUAL", "1") != "0"

# Fundamental screen thresholds (override via env vars).
MIN_REVENUE = float(os.getenv("MIN_REVENUE", "0"))
MIN_NET_INCOME = float(os.getenv("MIN_NET_INCOME", "0"))
MIN_EPS = float(os.getenv("MIN_EPS", "0"))
MAX_DEBT_TO_ASSETS = float(os.getenv("MAX_DEBT_TO_ASSETS", "0.8"))

EXCLUDE_SPECIAL_TICKERS = os.getenv("EXCLUDE_SPECIAL_TICKERS", "1") != "0"
EXCLUDE_SUFFIXES = [s.strip().upper() for s in os.getenv("EXCLUDE_SUFFIXES", "-W,-U,-R").split(",") if s.strip()]

if FAST_PROFILE:
    # Override defaults for aggressive speed (rate limit still enforced)
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

if SEC_USER_AGENT == "Your Name your.email@example.com":
    logging.warning(
        "SEC_USER_AGENT not set; SEC requires a descriptive User-Agent with contact info."
    )

_sec_lock = threading.Lock()
_last_sec_call_ts = [0.0]
_sec_ticker_map_cache = {"ts": 0.0, "data": None}
_sec_ticker_map_lock = threading.Lock()

_http_session = None


# ==============================
# Helpers
# ==============================
def map_symbols_for_providers(symbol: str):
    """
    Return (display_symbol, sec_ticker).
    SEC tickers use '-' for class shares like BRK-B; normalize dots to dashes.
    """
    display_symbol = symbol.strip().upper()
    sec_ticker = display_symbol.replace(".", "-")
    return display_symbol, sec_ticker


def get_http_session():
    global _http_session
    if _http_session is not None:
        return _http_session
    session = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _http_session = session
    return _http_session


# ==============================
# Data helpers
# ==============================


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


# ==============================
# SEC EDGAR helpers
# ==============================
class SecHTTPError(RuntimeError):
    def __init__(self, status_code, text):
        super().__init__(f"SEC HTTP {status_code}: {text}")
        self.status_code = status_code
        self.text = text


SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

ANNUAL_FORMS = {"10-K", "20-F", "40-F"}

REVENUE_KEYS = (
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueGoodsNet",
)
NET_INCOME_KEYS = ("NetIncomeLoss", "ProfitLoss")
ASSETS_KEYS = ("Assets",)
LIABILITIES_KEYS = ("Liabilities",)
EQUITY_KEYS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
EPS_KEYS = ("EarningsPerShareBasic", "EarningsPerShareDiluted")
SHARES_DEI_KEYS = ("EntityCommonStockSharesOutstanding",)
SHARES_GAAP_KEYS = ("CommonStockSharesOutstanding", "CommonStockSharesIssued")


def sec_request_url(url: str, params=None):
    session = get_http_session()
    with _sec_lock:
        now = time.time()
        wait_for = SEC_RATE_LIMIT_SECONDS - (now - _last_sec_call_ts[0])
        if wait_for > 0:
            time.sleep(wait_for)
        resp = session.get(url, params=params, headers=SEC_HEADERS, timeout=HTTP_TIMEOUT)
        _last_sec_call_ts[0] = time.time()
    if resp.status_code != 200:
        raise SecHTTPError(resp.status_code, resp.text)
    return resp.json()


def sec_request(path: str, params=None):
    url = f"{SEC_BASE_URL}{path}"
    return sec_request_url(url, params=params)


def _normalize_cik(cik):
    return f"{int(cik):010d}"


def _parse_sec_ticker_mapping(data):
    mapping = {}
    if isinstance(data, dict) and "data" in data and "fields" in data:
        fields = data.get("fields") or []
        for row in data.get("data") or []:
            if not isinstance(row, (list, tuple)):
                continue
            item = dict(zip(fields, row))
            ticker = item.get("ticker")
            cik = item.get("cik") or item.get("cik_str")
            if ticker and cik is not None:
                mapping[str(ticker).upper()] = int(cik)
        return mapping
    if isinstance(data, dict):
        for item in data.values():
            if not isinstance(item, dict):
                continue
            ticker = item.get("ticker")
            cik = item.get("cik_str") or item.get("cik")
            if ticker and cik is not None:
                mapping[str(ticker).upper()] = int(cik)
        return mapping
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            ticker = item.get("ticker")
            cik = item.get("cik_str") or item.get("cik")
            if ticker and cik is not None:
                mapping[str(ticker).upper()] = int(cik)
    return mapping


def fetch_sec_ticker_json():
    urls = [
        f"{SEC_BASE_URL}/api/xbrl/company_tickers.json",
        f"{SEC_BASE_URL}/files/company_tickers.json",
        f"{SEC_FILES_BASE_URL}/files/company_tickers.json",
        f"{SEC_FILES_BASE_URL}/files/company_tickers_exchange.json",
    ]
    last_err = None
    for url in urls:
        try:
            return sec_request_url(url)
        except SecHTTPError as err:
            last_err = err
            if err.status_code in (403, 404):
                logging.debug(f"SEC ticker list unavailable at {url} ({err.status_code}).")
                continue
            raise
    if last_err:
        raise last_err
    return None


def get_sec_ticker_cik_map():
    cache = _sec_ticker_map_cache.get("data")
    if cache and (time.time() - _sec_ticker_map_cache.get("ts", 0.0) < SEC_TICKER_CACHE_TTL):
        return cache
    with _sec_ticker_map_lock:
        cache = _sec_ticker_map_cache.get("data")
        if cache and (time.time() - _sec_ticker_map_cache.get("ts", 0.0) < SEC_TICKER_CACHE_TTL):
            return cache
        data = fetch_sec_ticker_json()
        mapping = _parse_sec_ticker_mapping(data or {})
        _sec_ticker_map_cache["data"] = mapping
        _sec_ticker_map_cache["ts"] = time.time()
        return mapping


def get_cik_for_ticker(ticker: str):
    if not ticker:
        return None
    mapping = get_sec_ticker_cik_map()
    return mapping.get(ticker.upper())


def fetch_sec_company_facts(cik: int):
    cik_str = _normalize_cik(cik)
    return sec_request(f"/api/xbrl/companyfacts/CIK{cik_str}.json")


def parse_sec_date(date_str):
    try:
        return datetime.date.fromisoformat(date_str)
    except Exception:
        return None


def extract_fact_entries(facts: dict, taxonomy: str, key: str):
    node = facts.get(taxonomy, {}).get(key)
    if not node:
        return []
    units = node.get("units", {})
    entries = []
    for unit, items in units.items():
        for item in items:
            val = item.get("val")
            date_str = item.get("end") or item.get("instant")
            dt = parse_sec_date(date_str) if date_str else None
            if dt is None or val is None:
                continue
            entries.append({
                "date": dt,
                "val": val,
                "unit": unit,
                "form": item.get("form"),
                "fy": item.get("fy"),
                "fp": item.get("fp"),
                "filed": item.get("filed"),
            })
    return entries


def pick_latest_entry(entries, preferred_units=None, prefer_annual=True):
    if not entries:
        return None
    if preferred_units:
        preferred = [e for e in entries if e["unit"] in preferred_units]
        if preferred:
            entries = preferred
    if prefer_annual:
        annual_entries = [e for e in entries if e.get("form") in ANNUAL_FORMS or e.get("fp") == "FY"]
        if annual_entries:
            entries = annual_entries
    entries_sorted = sorted(entries, key=lambda e: (e["date"], e.get("filed") or ""), reverse=True)
    return entries_sorted[0]


def find_fact_entry(facts: dict, taxonomy: str, keys, preferred_units=None, prefer_annual=True):
    for key in keys:
        entries = extract_fact_entries(facts, taxonomy, key)
        entry = pick_latest_entry(entries, preferred_units=preferred_units, prefer_annual=prefer_annual)
        if entry:
            entry = dict(entry)
            entry["key"] = key
            return entry
    return None


def compute_yoy_growth(entries, preferred_units=None):
    if not entries:
        return None
    if preferred_units:
        entries = [e for e in entries if e["unit"] in preferred_units]
    annual_entries = [e for e in entries if e.get("form") in ANNUAL_FORMS or e.get("fp") == "FY"]
    if len(annual_entries) < 2:
        return None
    by_year = {}
    for entry in annual_entries:
        year = entry["date"].year
        if year not in by_year or entry["date"] > by_year[year]["date"]:
            by_year[year] = entry
    years = sorted(by_year.keys(), reverse=True)
    if len(years) < 2:
        return None
    latest = by_year[years[0]]
    prior = by_year[years[1]]
    latest_val = coerce_first_numeric(latest.get("val"))
    prior_val = coerce_first_numeric(prior.get("val"))
    if latest_val is None or prior_val is None or prior_val == 0:
        return None
    return (latest_val - prior_val) / abs(prior_val) * 100


# ==============================
# Ticker universe
# ==============================
def save_normalized_tickers_csv(input_csv="nyse_tickers.csv", output_csv=None):
    """
    Normalize tickers from a local CSV and save to a new CSV.
    """
    tickers = load_tickers_from_csv(input_csv)
    if output_csv is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"nyse_tickers_normalized_{ts}.csv"
    pd.DataFrame({"Ticker": tickers}).to_csv(output_csv, index=False)
    return output_csv, len(tickers)


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
    if EXCLUDE_SPECIAL_TICKERS and tickers:
        before = len(tickers)
        tickers = [
            t for t in tickers
            if "$" not in t and not any(t.endswith(suf) for suf in EXCLUDE_SUFFIXES)
        ]
        removed = before - len(tickers)
        if removed:
            logging.info(f"Excluded {removed} special tickers (preferreds/warrants/units).")
    if not tickers:
        raise ValueError(f"No tickers parsed from CSV {csv_path}.")
    return tickers


# ==============================
# Analyze a single ticker
# ==============================
def analyze_ticker(symbol, facts=None, ignore_thresholds=False):
    try:
        display_symbol, sec_ticker = map_symbols_for_providers(symbol)
        cik = get_cik_for_ticker(sec_ticker)
        if cik is None:
            logging.debug(f"No CIK mapping for {display_symbol}.")
            return None
        if facts is None:
            facts = fetch_sec_company_facts(cik)
        if not facts or "facts" not in facts:
            logging.debug(f"No SEC company facts for {display_symbol}.")
            return None

        fact_data = facts.get("facts", {})
        entity_name = facts.get("entityName")
        cik_str = _normalize_cik(cik)

        revenue_entry = find_fact_entry(
            fact_data, "us-gaap", REVENUE_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
        )
        net_income_entry = find_fact_entry(
            fact_data, "us-gaap", NET_INCOME_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
        )
        assets_entry = find_fact_entry(
            fact_data, "us-gaap", ASSETS_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
        )
        liabilities_entry = find_fact_entry(
            fact_data, "us-gaap", LIABILITIES_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
        )
        equity_entry = find_fact_entry(
            fact_data, "us-gaap", EQUITY_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
        )
        eps_entry = find_fact_entry(
            fact_data, "us-gaap", EPS_KEYS, preferred_units=("USD/shares",), prefer_annual=SEC_PREFER_ANNUAL
        )
        shares_entry = find_fact_entry(
            fact_data, "dei", SHARES_DEI_KEYS, preferred_units=("shares",), prefer_annual=False
        )
        if shares_entry is None:
            shares_entry = find_fact_entry(
                fact_data, "us-gaap", SHARES_GAAP_KEYS, preferred_units=("shares",), prefer_annual=False
            )

        revenue_val = coerce_first_numeric(revenue_entry["val"]) if revenue_entry else None
        net_income_val = coerce_first_numeric(net_income_entry["val"]) if net_income_entry else None
        assets_val = coerce_first_numeric(assets_entry["val"]) if assets_entry else None
        liabilities_val = coerce_first_numeric(liabilities_entry["val"]) if liabilities_entry else None
        equity_val = coerce_first_numeric(equity_entry["val"]) if equity_entry else None
        eps_val = coerce_first_numeric(eps_entry["val"]) if eps_entry else None
        shares_out_val = coerce_first_numeric(shares_entry["val"]) if shares_entry else None

        net_margin_pct = (
            (net_income_val / revenue_val) * 100
            if revenue_val not in (None, 0) and net_income_val is not None
            else None
        )
        debt_assets_ratio = (
            (liabilities_val / assets_val)
            if assets_val not in (None, 0) and liabilities_val is not None
            else None
        )
        debt_assets_pct = debt_assets_ratio * 100 if debt_assets_ratio is not None else None

        revenue_yoy_pct = None
        if revenue_entry:
            revenue_entries = extract_fact_entries(fact_data, "us-gaap", revenue_entry["key"])
            revenue_yoy_pct = compute_yoy_growth(revenue_entries, preferred_units=("USD",))

        report_dates = [
            entry["date"]
            for entry in (revenue_entry, net_income_entry, assets_entry, liabilities_entry, equity_entry)
            if entry
        ]
        report_date = max(report_dates).isoformat() if report_dates else None

        # Apply thresholds if not ignoring
        if not ignore_thresholds:
            if revenue_val is None or net_income_val is None:
                return None
            if revenue_val < MIN_REVENUE or net_income_val < MIN_NET_INCOME:
                return None
            if eps_val is None or eps_val < MIN_EPS:
                return None
            if debt_assets_ratio is None or debt_assets_ratio > MAX_DEBT_TO_ASSETS:
                return None

        return {
            "Ticker": display_symbol,
            "Company": entity_name,
            "CIK": cik_str,
            "ReportDate": report_date,
            "Revenue": int(revenue_val) if revenue_val is not None else None,
            "NetIncome": int(net_income_val) if net_income_val is not None else None,
            "NetMargin%": round(float(net_margin_pct), 2) if net_margin_pct is not None else None,
            "RevenueYoY%": round(float(revenue_yoy_pct), 2) if revenue_yoy_pct is not None else None,
            "Assets": int(assets_val) if assets_val is not None else None,
            "Liabilities": int(liabilities_val) if liabilities_val is not None else None,
            "Equity": int(equity_val) if equity_val is not None else None,
            "Debt/Assets%": round(float(debt_assets_pct), 2) if debt_assets_pct is not None else None,
            "EPS": round(float(eps_val), 2) if eps_val is not None else None,
            "SharesOut": int(shares_out_val) if shares_out_val is not None else None,
        }

    except Exception:
        logging.exception(f"Ticker {symbol} error")
    return None


def diagnose_ticker(symbol):
    """
    Print high-level reasons when a single ticker cannot be analyzed.
    """
    display_symbol, sec_ticker = map_symbols_for_providers(symbol)
    cik = get_cik_for_ticker(sec_ticker)
    if cik is None:
        logging.info(f"{display_symbol}: no CIK found in SEC ticker list.")
        return
    try:
        facts = fetch_sec_company_facts(cik)
    except SecHTTPError as err:
        logging.info(f"{display_symbol}: SEC request failed ({err.status_code}).")
        return
    if not facts or "facts" not in facts:
        logging.info(f"{display_symbol}: no SEC company facts returned.")
        return
    fact_data = facts.get("facts", {})
    logging.info(f"{display_symbol}: CIK={_normalize_cik(cik)}, entity={facts.get('entityName')}")

    revenue_entry = find_fact_entry(
        fact_data, "us-gaap", REVENUE_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
    )
    net_income_entry = find_fact_entry(
        fact_data, "us-gaap", NET_INCOME_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
    )
    assets_entry = find_fact_entry(
        fact_data, "us-gaap", ASSETS_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
    )
    liabilities_entry = find_fact_entry(
        fact_data, "us-gaap", LIABILITIES_KEYS, preferred_units=("USD",), prefer_annual=SEC_PREFER_ANNUAL
    )

    if not revenue_entry:
        logging.info(f"{display_symbol}: revenue fact not found.")
    else:
        logging.info(
            f"{display_symbol}: revenue={revenue_entry['val']} {revenue_entry['unit']} "
            f"({revenue_entry['date']})"
        )
    if not net_income_entry:
        logging.info(f"{display_symbol}: net income fact not found.")
    else:
        logging.info(
            f"{display_symbol}: net_income={net_income_entry['val']} {net_income_entry['unit']} "
            f"({net_income_entry['date']})"
        )
    if not assets_entry or not liabilities_entry:
        logging.info(f"{display_symbol}: assets/liabilities facts missing.")
    else:
        logging.info(
            f"{display_symbol}: assets={assets_entry['val']} liabilities={liabilities_entry['val']}"
        )


# ==============================
# Run screener
# ==============================
def run_screener(tickers):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(analyze_ticker, t, None): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            symbol = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logging.debug(f"Error for {symbol}: {e}")
            time.sleep(0.01)  # small delay to reduce burstiness

    df = pd.DataFrame(results)

    if df.empty:
        logging.info("No candidates matched filters this run.")
        return df

    if "Revenue" in df.columns:
        df = df.sort_values(by="Revenue", ascending=False, na_position="last")
    elif "NetMargin%" in df.columns:
        df = df.sort_values(by="NetMargin%", ascending=False, na_position="last")

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"candidates_{ts}.csv"
    with open(out_path, 'w') as f:
        f.write("# Data source: SEC EDGAR (data.sec.gov)\n")
        f.write("# Screener Parameters:\n")
        f.write(f"# Revenue >= {MIN_REVENUE}\n")
        f.write(f"# NetIncome >= {MIN_NET_INCOME}\n")
        f.write(f"# EPS >= {MIN_EPS}\n")
        f.write(f"# Debt/Assets <= {MAX_DEBT_TO_ASSETS}\n\n")
    df.to_csv(out_path, mode='a', index=False)
    logging.info(f"Found {len(df)} candidates. Saved to {out_path}")
    return df


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    print("Select mode:")
    print("1 - Scan all NYSE tickers (from nyse_tickers.csv)")
    print("2 - Scan a specific ticker")
    print("3 - Normalize nyse_tickers.csv and save a clean copy")
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
                f.write("# Data source: SEC EDGAR (data.sec.gov)\n")
                f.write("# Screener Parameters:\n")
                f.write(f"# Revenue >= {MIN_REVENUE}\n")
                f.write(f"# NetIncome >= {MIN_NET_INCOME}\n")
                f.write(f"# EPS >= {MIN_EPS}\n")
                f.write(f"# Debt/Assets <= {MAX_DEBT_TO_ASSETS}\n\n")
            df.to_csv(out_path, mode='a', index=False)
            logging.info(f"Saved single ticker result to {out_path}")
        else:
            logging.info(f"Could not fetch metrics for {ticker_input}.")
            diagnose_ticker(ticker_input)

    elif choice == "3":
        out_path, count = save_normalized_tickers_csv("nyse_tickers.csv")
        logging.info(f"Saved {count} normalized tickers to {out_path}")

    else:
        print("Invalid choice. Exiting.")
        exit()

    if not df.empty:
        print(df.head(30))
