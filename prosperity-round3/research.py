from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")

PRICE_FILES = [
    DATA_DIR / "prices_round_0_day_-2.csv",
    DATA_DIR / "prices_round_0_day_-1.csv",
]

TRADE_FILES = [
    DATA_DIR / "trades_round_0_day_-2.csv",
    DATA_DIR / "trades_round_0_day_-1.csv",
]


def load_csvs(files):
    frames = []
    for file in files:
        df = pd.read_csv(file, sep=";")
        df["source_file"] = file.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()

    # best bid / best ask aus Level 1
    prices["best_bid"] = prices["bid_price_1"]
    prices["best_ask"] = prices["ask_price_1"]

    # Midprice und Spread
    prices["spread"] = prices["best_ask"] - prices["best_bid"]
    prices["mid"] = (prices["best_ask"] + prices["best_bid"]) / 2

    # Midprice-Veränderung
    prices["mid_change"] = prices.groupby("product")["mid"].diff()

    return prices


def summarize_product(prices: pd.DataFrame, trades: pd.DataFrame, product: str):
    p = prices[prices["product"] == product].copy()
    t = trades[trades["symbol"] == product].copy()

    print("\n" + "=" * 60)
    print(f"PRODUKT: {product}")
    print("=" * 60)

    print(f"Anzahl Preiszeilen: {len(p)}")
    print(f"Anzahl Trades: {len(t)}")
    print()

    print("PREISSTATISTIK")
    print(f"Best Bid min/max: {p['best_bid'].min()} / {p['best_bid'].max()}")
    print(f"Best Ask min/max: {p['best_ask'].min()} / {p['best_ask'].max()}")
    print(f"Mid min/max: {p['mid'].min()} / {p['mid'].max()}")
    print(f"Mid Mittelwert: {p['mid'].mean():.4f}")
    print(f"Mid Standardabweichung: {p['mid'].std():.4f}")
    print()

    print("SPREAD")
    print(f"Spread min/max: {p['spread'].min()} / {p['spread'].max()}")
    print(f"Spread Mittelwert: {p['spread'].mean():.4f}")
    print(f"Spread Median: {p['spread'].median():.4f}")
    print()

    print("MID-VERÄNDERUNG")
    print(f"Durchschnittliche Mid-Änderung: {p['mid_change'].mean():.6f}")
    print(f"Std der Mid-Änderung: {p['mid_change'].std():.6f}")
    print()

    if len(t) > 0:
        print("TRADE-STATISTIK")
        print(f"Trade-Preis min/max: {t['price'].min()} / {t['price'].max()}")
        print(f"Trade-Preis Mittelwert: {t['price'].mean():.4f}")
        print(f"Durchschnittliche Trade-Größe: {t['quantity'].mean():.4f}")
        print(f"Gesamtes gehandeltes Volumen: {t['quantity'].sum()}")
    else:
        print("Keine Trades vorhanden.")


def main():
    prices = load_csvs(PRICE_FILES)
    trades = load_csvs(TRADE_FILES)

    prices = prepare_prices(prices)

    print("Produkte in prices:", sorted(prices["product"].dropna().unique()))
    print("Produkte in trades:", sorted(trades["symbol"].dropna().unique()))

    for product in sorted(prices["product"].dropna().unique()):
        summarize_product(prices, trades, product)


if __name__ == "__main__":
    main()