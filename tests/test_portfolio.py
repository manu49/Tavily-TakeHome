"""Tests for portfolio.py — upload parsing, validation, and the 30-day store. Offline:
file contents are built as bytes in-memory (no real upload, no network)."""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest

import portfolio as pf


# --------------------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------------------

class TestParseCSV:
    def test_percentage_weights_normalize_to_fractions(self):
        result = pf.parse_portfolio(b"ticker,weight\nAAPL,60%\nMSFT,40%", "p.csv")
        assert result.ok
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})

    def test_bare_hundred_scale_weights_treated_as_percent(self):
        result = pf.parse_portfolio(b"ticker,weight\nAAPL,60\nMSFT,40", "p.csv")
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})

    def test_fractional_weights(self):
        result = pf.parse_portfolio(b"ticker,weight\nAAPL,0.7\nMSFT,0.3", "p.csv")
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.7, "MSFT": 0.3})

    def test_flexible_headers(self):
        result = pf.parse_portfolio(b"Symbol,Allocation\nAAPL,50%\nGOOG,50%", "p.csv")
        assert set(result.portfolio.tickers()) == {"AAPL", "GOOG"}

    def test_quantities_only_flags_derived_weights(self):
        result = pf.parse_portfolio(b"ticker,shares\nAAPL,10\nMSFT,5", "p.csv")
        assert result.ok
        assert result.portfolio.has_quantities()
        assert result.portfolio.normalized_weights() is None  # need prices to derive
        assert any("derived" in w for w in result.warnings)

    def test_ticker_only_is_equal_weighted(self):
        result = pf.parse_portfolio(b"ticker\nAAPL\nMSFT\nGOOG", "p.csv")
        assert result.portfolio.normalized_weights() == pytest.approx(
            {"AAPL": 1 / 3, "MSFT": 1 / 3, "GOOG": 1 / 3}
        )

    def test_no_ticker_header_uses_first_column(self):
        result = pf.parse_portfolio(b"name,weight\nAAPL,1.0", "p.csv")
        assert result.portfolio.tickers() == ["AAPL"]
        assert any("ticker column" in w for w in result.warnings)

    def test_weights_not_summing_to_one_warns(self):
        result = pf.parse_portfolio(b"ticker,weight\nAAPL,0.6\nMSFT,0.6", "p.csv")
        assert any("normalized" in w for w in result.warnings)
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.5, "MSFT": 0.5})

    def test_duplicate_tickers_merged(self):
        result = pf.parse_portfolio(b"ticker,weight\nAAPL,0.3\nAAPL,0.3\nMSFT,0.4", "p.csv")
        assert any("Duplicate" in w for w in result.warnings)
        assert set(result.portfolio.tickers()) == {"AAPL", "MSFT"}

    def test_unrecognized_ticker_skipped(self):
        result = pf.parse_portfolio(b"ticker,weight\nAAPL,0.5\n!!!,0.5", "p.csv")
        assert result.portfolio.tickers() == ["AAPL"]

    def test_dollar_amount_column_drives_weights(self):
        # The reported case: a "Dollar Amount" column must yield value-proportional weights,
        # not equal weight. 300M / 304M and 4M / 304M.
        result = pf.parse_portfolio(b"Ticker,Dollar Amount\nAAPL,300000000\nSPCX,4000000", "p.csv")
        assert result.ok
        w = result.portfolio.normalized_weights()
        assert w["AAPL"] == pytest.approx(300 / 304, abs=1e-4)
        assert w["SPCX"] == pytest.approx(4 / 304, abs=1e-4)
        assert any("market-value" in x for x in result.warnings)

    def test_market_value_header_variant(self):
        result = pf.parse_portfolio(b"symbol,Market Value\nAAPL,750\nMSFT,250", "p.csv")
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.75, "MSFT": 0.25})

    def test_tab_separated_with_dollar_amount(self):
        result = pf.parse_portfolio(b"Ticker\tDollar Amount\nAAPL\t300000000\nSPCX\t4000000", "p.csv")
        assert result.ok
        assert result.portfolio.normalized_weights()["AAPL"] == pytest.approx(300 / 304, abs=1e-4)


# --------------------------------------------------------------------------------------
# Excel parsing
# --------------------------------------------------------------------------------------

class TestParseExcel:
    def test_xlsx_round_trip(self):
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight": [0.6, 0.4]})
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        result = pf.parse_portfolio(buf.getvalue(), "holdings.xlsx")
        assert result.ok
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})


# --------------------------------------------------------------------------------------
# Text parsing
# --------------------------------------------------------------------------------------

class TestParseText:
    def test_percent_lines(self):
        result = pf.parse_portfolio(b"AAPL 60%\nMSFT 40%", "p.txt")
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})

    def test_comma_separated(self):
        result = pf.parse_portfolio(b"AAPL, 0.5\nMSFT, 0.5", "p.txt")
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.5, "MSFT": 0.5})

    def test_large_bare_numbers_are_quantities(self):
        result = pf.parse_portfolio(b"AAPL 100\nMSFT 50", "p.txt")
        assert result.portfolio.has_quantities()

    def test_comments_and_junk_skipped(self):
        result = pf.parse_portfolio(b"# my portfolio\nAAPL 50%\n\nnot a ticker line!!\nMSFT 50%", "p.txt")
        assert set(result.portfolio.tickers()) == {"AAPL", "MSFT"}

    def test_bare_ticker_list_equal_weight(self):
        result = pf.parse_portfolio(b"AAPL\nMSFT", "p.txt")
        assert result.portfolio.normalized_weights() == pytest.approx({"AAPL": 0.5, "MSFT": 0.5})


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------

class TestParseErrors:
    def test_empty_file(self):
        result = pf.parse_portfolio(b"   ", "p.csv")
        assert not result.ok
        assert result.errors

    def test_no_valid_holdings(self):
        result = pf.parse_portfolio(b"ticker,weight\n!!!,0.5\n???,0.5", "p.csv")
        assert not result.ok

    def test_report_is_serializable(self):
        import json
        result = pf.parse_portfolio(b"AAPL 50%\nMSFT 50%", "p.txt")
        json.dumps(result.report())  # must not raise


# --------------------------------------------------------------------------------------
# Portfolio model
# --------------------------------------------------------------------------------------

class TestPortfolioModel:
    def test_round_trip_dict(self):
        p = pf.Portfolio([pf.Holding("AAPL", weight=0.6), pf.Holding("MSFT", weight=0.4)])
        restored = pf.Portfolio.from_dict(p.to_dict())
        assert restored.tickers() == ["AAPL", "MSFT"]
        assert restored.normalized_weights() == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})


# --------------------------------------------------------------------------------------
# PortfolioStore (30-day TTL)
# --------------------------------------------------------------------------------------

class TestPortfolioStore:
    def test_save_and_load_round_trip(self, tmp_path):
        store = pf.PortfolioStore(tmp_path)
        p = pf.Portfolio([pf.Holding("AAPL", weight=1.0)])
        pid = store.save(p)
        loaded = store.load(pid)
        assert loaded is not None and loaded.tickers() == ["AAPL"]

    def test_missing_id_returns_none(self, tmp_path):
        assert pf.PortfolioStore(tmp_path).load("nope") is None

    def test_expired_entry_is_none_and_purged(self, tmp_path):
        store = pf.PortfolioStore(tmp_path)
        old = pf.Portfolio([pf.Holding("AAPL", weight=1.0)], created_at=time.time() - 40 * 24 * 3600)
        pid = store.save(old)
        assert store.load(pid) is None             # older than 30-day TTL
        assert not store._path(pid).exists()        # purged on access

    def test_purge_expired_counts_removed(self, tmp_path):
        store = pf.PortfolioStore(tmp_path)
        store.save(pf.Portfolio([pf.Holding("AAPL", weight=1.0)], created_at=time.time() - 40 * 24 * 3600))
        store.save(pf.Portfolio([pf.Holding("MSFT", weight=1.0)]))  # fresh
        assert store.purge_expired() == 1
