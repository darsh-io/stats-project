"""Sector portfolio backtest.

Simulates investing a fixed dollar amount in each ticker on its start date,
grouped into sector baskets, and writes the resulting daily portfolio value
to CSV.

Usage:
    python portfolio_backtest.py
    python portfolio_backtest.py --investment 50000 --start-date 2010-01-01 --output out.csv
    python portfolio_backtest.py --no-cache
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
    
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    investment_per_company: float = 20_000.0
    start_date: str = "2005-01-04"
    output_path: str = "DATA.csv"
    cache_dir: str = ".cache"
    use_cache: bool = True


# Equal-dollar-per-stock sector baskets. Edit here to add/remove tickers.
DEFAULT_SECTORS = {
    "Tech": [
        "AAPL",   # Consumer tech
        "MSFT",   # Software & cloud
        "GOOG",  # Internet/search
        "NVDA",   # AI & GPUs
        "ORCL",   # Enterprise software
        "INTC",   # CPUs
        "AMD",    # CPUs & GPUs
        "AMZN",   # E-commerce & cloud
        "CSCO",   # Networking
        "ADBE",   # Creative software
        "QCOM",   # Wireless & mobile chips
        "IBM",    # Enterprise & consulting
    ]
}


# --------------------------------------------------------------------------
# Data fetching (isolates all yfinance interaction + caching)
# --------------------------------------------------------------------------

class StockDataFetcher:
    """Downloads a single ticker's Close series, with optional on-disk caching."""

    def __init__(self, start_date: str, cache_dir: Optional[str] = None):
        self.start_date = start_date
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_{self.start_date}.csv"

    def get_close_series(self, ticker: str) -> Optional[pd.Series]:
        """Returns a Close price Series indexed by date, or None on failure.

        Returns None (rather than raising) on delisted tickers, network
        failures, or empty responses so callers can skip a bad ticker
        without crashing the whole run.
        """
        cache_file = self._cache_path(ticker) if self.cache_dir else None
        if cache_file and cache_file.exists():
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return cached["Close"]

        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:  # yfinance raises assorted exception types
            logger.warning("Download failed for %s: %s", ticker, exc)
            return None

        # yfinance returns MultiIndex columns (Field, Ticker) even for a
        # single ticker on current versions -- flatten defensively so this
        # keeps working whichever shape a given yfinance version returns.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty or "Close" not in data.columns:
            logger.warning("No data returned for %s", ticker)
            return None

        if cache_file:
            data.to_csv(cache_file)

        return data["Close"]


# --------------------------------------------------------------------------
# Portfolio construction (pure computation, no I/O)
# --------------------------------------------------------------------------

class PortfolioBuilder:
    def __init__(self, fetcher: StockDataFetcher, investment_per_company: float):
        self.fetcher = fetcher
        self.investment_per_company = investment_per_company

    def _build_sector_value(self, sector: str, tickers: List[str]) -> pd.Series:
        position_values: Dict[str, pd.Series] = {}
        for ticker in tickers:
            close = self.fetcher.get_close_series(ticker)
            if close is None or close.empty:
                logger.warning("Skipping %s (sector %s): no usable data", ticker, sector)
                continue
            shares = self.investment_per_company / close.iloc[0]
            position_values[ticker] = close * shares

        if not position_values:
            logger.warning("Sector %s has no valid tickers; returning empty series", sector)
            return pd.Series(dtype=float)

        combined = pd.concat(position_values, axis=1)

        # Bug fix vs. a naive sum(): if a ticker's history starts later than
        # its sector-mates (e.g. a 2008 IPO vs. a 2005 start date), pandas'
        # default sum(skipna=True) silently treats its missing early rows as
        # 0 instead of "not yet invested." That understates the sector total
        # for years, then produces a fake return spike the day the late
        # ticker's data begins. Restricting to rows where every ticker in
        # the sector has a real price removes that artifact.
        aligned = combined.dropna()
        if len(aligned) < len(combined):
            logger.info(
                "Sector %s: trimmed %d/%d rows lacking full ticker coverage "
                "(staggered start dates)",
                sector, len(combined) - len(aligned), len(combined),
            )

        return aligned.sum(axis=1)

    def build(self, sectors: Dict[str, List[str]]) -> pd.DataFrame:
        portfolio = pd.DataFrame({
            sector: self._build_sector_value(sector, tickers)
            for sector, tickers in sectors.items()
        })
        portfolio["Total Portfolio"] = portfolio.sum(axis=1)
        return portfolio


# --------------------------------------------------------------------------
# CLI / orchestration
# --------------------------------------------------------------------------

def parse_args() -> BacktestConfig:
    defaults = BacktestConfig()
    parser = argparse.ArgumentParser(description="Sector portfolio backtest")
    parser.add_argument("--investment", type=float, default=defaults.investment_per_company,
                         help="Dollars invested per company at its start date")
    parser.add_argument("--start-date", type=str, default=defaults.start_date,
                         help="YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=defaults.output_path)
    parser.add_argument("--no-cache", action="store_true",
                         help="Force re-download instead of using cached CSVs")
    args = parser.parse_args()

    return BacktestConfig(
        investment_per_company=args.investment,
        start_date=args.start_date,
        output_path=args.output,
        use_cache=not args.no_cache,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_args()

    fetcher = StockDataFetcher(
        start_date=config.start_date,
        cache_dir=config.cache_dir if config.use_cache else None,
    )
    builder = PortfolioBuilder(fetcher, config.investment_per_company)
    portfolio = builder.build(DEFAULT_SECTORS)

    portfolio.to_csv(config.output_path)
    logging.info("Saved portfolio to %s (%d rows, %d columns)",
                 config.output_path, len(portfolio), len(portfolio.columns))


if __name__ == "__main__":
    main()
