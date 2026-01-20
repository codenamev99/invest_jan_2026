import numpy as np
import pandas as pd

import main


def test_map_symbols_for_providers_handles_dot_symbols():
    yahoo, av = main.map_symbols_for_providers("BRK.B")
    assert yahoo == "BRK-B"
    assert av == "BRK.B"


def test_coerce_first_numeric_handles_sequences_and_nan():
    assert main.coerce_first_numeric([1, 2, 3]) == 1.0
    assert main.coerce_first_numeric(np.array([2])) == 2.0
    assert main.coerce_first_numeric(pd.Series([3])) == 3.0
    assert main.coerce_first_numeric("x") is None
    assert main.coerce_first_numeric(np.nan) is None


def test_normalize_ohlc_columns_flattens_multiindex():
    columns = pd.MultiIndex.from_tuples(
        [("AAPL", "Open"), ("AAPL", "Close"), ("AAPL", "Volume")]
    )
    hist = pd.DataFrame([[1, 2, 3]], columns=columns)
    normalized = main.normalize_ohlc_columns(hist)
    assert list(normalized.columns) == ["Open", "Close", "Volume"]


def test_get_rsi_wilder_local_returns_series_in_range():
    prices = pd.Series(np.linspace(100, 120, 40))
    hist = pd.DataFrame({"Close": prices})
    rsi = main.get_rsi_wilder_local(hist, period=14, use_adj_close=False)
    assert len(rsi) == len(prices)
    assert rsi.iloc[-1] >= 0
    assert rsi.iloc[-1] <= 100
