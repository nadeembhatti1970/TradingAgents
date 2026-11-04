#!/usr/bin/env python3
"""Look up the Trading 212 instrument ticker for one or more Yahoo-style symbols.

Read-only: it only calls GET /equity/metadata/instruments. Matching uses the same logic as the app (tradingagents.brokers.trading212).
Credentials come from
T212_API_KEY / T212_API_SECRET (or a .env file in the current directory).

    python scripts/t212_lookup.py HSBA.L SHEL.L AZN.L
    python scripts/t212_lookup.py --mode real NVDA
    python scripts/t212_lookup.py --list-etf ISF.L --top 10

Trading 212 allows this endpoint about once per 50 seconds, so every symbol is
looked up from a single download.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.brokers.trading212 import find_instrument_matches  # noqa: E402

BASE_URLS = {
    "practice": "https://demo.trading212.com/api/v0",
    "real": "https://live.trading212.com/api/v0",
}

def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #")[0].strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def fetch_instruments(mode: str) -> list[dict]:
    key, secret = os.environ.get("T212_API_KEY"), os.environ.get("T212_API_SECRET")
    if not key or not secret:
        sys.exit("Set T212_API_KEY and T212_API_SECRET (environment or .env).")
    response = requests.get(
        f"{BASE_URLS[mode]}/equity/metadata/instruments", auth=(key, secret), timeout=60
    )
    if response.status_code == 429:
        sys.exit("Rate limited (this endpoint allows ~1 request per 50s). Retry shortly.")
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        sys.exit("Unexpected response: instruments payload was not a list.")
    return data


def describe(item: dict) -> str:
    return (
        f"{item.get('ticker')}  | {item.get('name')} | shortName={item.get('shortName')} "
        f"| {item.get('currencyCode')} | {item.get('isin')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("symbols", nargs="*", help="e.g. HSBA.L NVDA 0700.HK")
    parser.add_argument("--mode", choices=BASE_URLS, default="practice")
    parser.add_argument("--list-etf", metavar="ETF", help="look up that ETF's top holdings")
    parser.add_argument("--top", type=int, default=10, help="holdings to use with --list-etf")
    args = parser.parse_args()

    symbols = list(args.symbols)
    if args.list_etf:
        from tradingagents.etf import top_holdings

        symbols += top_holdings(args.list_etf, args.top)
    if not symbols:
        parser.error("give at least one symbol or --list-etf")

    load_dotenv()
    instruments = fetch_instruments(args.mode)
    print(f"{len(instruments)} instruments from the {args.mode} API\n")

    missing = 0
    for symbol in symbols:
        matches = find_instrument_matches(symbol, instruments)
        print(symbol)
        if not matches:
            missing += 1
            print("  no match (not tradable on this account, or the name differs)")
        for item in matches:
            print(f"  {describe(item)}")
        if len(matches) > 1:
            print("  note: several matches; the app needs exactly one")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
