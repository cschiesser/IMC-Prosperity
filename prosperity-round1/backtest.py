from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from datamodel import OrderDepth, Trade, TradingState, Observation
from trader import Trader


DATA_DIR = Path("data")
PRICE_FILES = [
    DATA_DIR / "prices_round_0_day_-2.csv",
    DATA_DIR / "prices_round_0_day_-1.csv",
]

TRADE_FILES = [
    DATA_DIR / "trades_round_0_day_-2.csv",
    DATA_DIR / "trades_round_0_day_-1.csv",
]

PRODUCTS = ["EMERALDS", "TOMATOES"]
FAIR_VALUE_EMERALDS = 10000


def load_csvs(files: List[Path]) -> pd.DataFrame:
    frames = []
    for file in files:
        df = pd.read_csv(file, sep=";")
        df["source_file"] = file.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


def build_order_depth(row: pd.Series) -> OrderDepth:
    order_depth = OrderDepth()

    for level in [1, 2, 3]:
        bid_price = safe_int(row.get(f"bid_price_{level}"))
        bid_volume = safe_int(row.get(f"bid_volume_{level}"))
        if bid_price is not None and bid_volume is not None and bid_volume > 0:
            order_depth.buy_orders[bid_price] = bid_volume

    for level in [1, 2, 3]:
        ask_price = safe_int(row.get(f"ask_price_{level}"))
        ask_volume = safe_int(row.get(f"ask_volume_{level}"))
        if ask_price is not None and ask_volume is not None and ask_volume > 0:
            order_depth.sell_orders[ask_price] = -ask_volume

    return order_depth


def build_market_trades(
    trades_df: pd.DataFrame,
    source_file: str,
    timestamp: int,
    product: str,
) -> List[Trade]:
    step_trades = trades_df[
        (trades_df["source_file"] == source_file)
        & (trades_df["timestamp"] == timestamp)
        & (trades_df["symbol"] == product)
    ]

    result = []
    for _, row in step_trades.iterrows():
        buyer = "" if pd.isna(row["buyer"]) else str(row["buyer"])
        seller = "" if pd.isna(row["seller"]) else str(row["seller"])

        result.append(
            Trade(
                symbol=str(row["symbol"]),
                price=int(row["price"]),
                quantity=int(row["quantity"]),
                buyer=buyer,
                seller=seller,
                timestamp=int(row["timestamp"]),
            )
        )

    return result


def enforce_position_limits(orders: List, current_position: int, limit: int) -> List:
    """
    Vereinfachte Prosperity-nahe Logik:
    Wenn aggregierte BUY-Menge das Long-Limit reißen würde, werden alle BUY-Orders verworfen.
    Wenn aggregierte SELL-Menge das Short-Limit reißen würde, werden alle SELL-Orders verworfen.
    """
    total_buy = sum(max(0, o.quantity) for o in orders)
    total_sell = sum(max(0, -o.quantity) for o in orders)

    allow_buys = current_position + total_buy <= limit
    allow_sells = current_position - total_sell >= -limit

    filtered = []
    for o in orders:
        if o.quantity > 0 and not allow_buys:
            continue
        if o.quantity < 0 and not allow_sells:
            continue
        filtered.append(o)

    return filtered


def simulate_immediate_fills(
    product: str,
    orders: List,
    order_depth: OrderDepth,
    timestamp: int,
) -> Tuple[List[Trade], int, List[Tuple[str, int, int]]]:
    """
    Führt sofort marketable Orders gegen das aktuelle sichtbare Buch aus.
    Rückgabe:
    - fills
    - cash_delta
    - residual_orders: Liste von (symbol, price, remaining_quantity)
      positive quantity = residual buy
      negative quantity = residual sell
    """
    fills: List[Trade] = []
    cash_delta = 0
    residual_orders: List[Tuple[str, int, int]] = []

    buy_book = dict(order_depth.buy_orders)
    sell_book = dict(order_depth.sell_orders)

    for order in orders:
        remaining = order.quantity

        # BUY ORDER
        if remaining > 0:
            for ask_price in sorted(sell_book.keys()):
                if remaining <= 0:
                    break
                if ask_price > order.price:
                    break

                available = -sell_book[ask_price]
                if available <= 0:
                    continue

                traded = min(remaining, available)
                if traded > 0:
                    fills.append(
                        Trade(
                            symbol=product,
                            price=ask_price,
                            quantity=traded,
                            buyer="SUBMISSION",
                            seller="",
                            timestamp=timestamp,
                        )
                    )
                    cash_delta -= ask_price * traded
                    remaining -= traded
                    sell_book[ask_price] += traded

            if remaining > 0:
                residual_orders.append((order.symbol, order.price, remaining))

        # SELL ORDER
        elif remaining < 0:
            remaining_to_sell = -remaining

            for bid_price in sorted(buy_book.keys(), reverse=True):
                if remaining_to_sell <= 0:
                    break
                if bid_price < order.price:
                    break

                available = buy_book[bid_price]
                if available <= 0:
                    continue

                traded = min(remaining_to_sell, available)
                if traded > 0:
                    fills.append(
                        Trade(
                            symbol=product,
                            price=bid_price,
                            quantity=traded,
                            buyer="",
                            seller="SUBMISSION",
                            timestamp=timestamp,
                        )
                    )
                    cash_delta += bid_price * traded
                    remaining_to_sell -= traded
                    buy_book[bid_price] -= traded

            if remaining_to_sell > 0:
                residual_orders.append((order.symbol, order.price, -remaining_to_sell))

    return fills, cash_delta, residual_orders


def simulate_passive_fills(
    residual_orders: List[Tuple[str, int, int]],
    current_order_depth: OrderDepth,
    next_order_depth: OrderDepth,
    next_market_trades: List[Trade],
    next_timestamp: int,
    fill_fraction: float = 0.25,
) -> Tuple[List[Trade], int]:
    """
    Verbesserte passive Fill-Approximation.

    Logik:
    1) Wenn das nächste sichtbare Buch eure Quote direkt kreuzt, füllen wir zuerst dagegen.
    2) Wenn eure Quote inside the spread liegt und das aktuelle Buch verbessert,
       dann nehmen wir bei Handelsaktivität im nächsten Schritt einen Teil-Fill an.
    3) Ausführungspreis = eure eigene Quote.
    """
    fills: List[Trade] = []
    cash_delta = 0

    current_best_bid = max(current_order_depth.buy_orders.keys()) if current_order_depth.buy_orders else None
    current_best_ask = min(current_order_depth.sell_orders.keys()) if current_order_depth.sell_orders else None
    next_total_trade_volume = sum(t.quantity for t in next_market_trades)

    for symbol, price, quantity in residual_orders:
        # PASSIVE BUY
        if quantity > 0:
            remaining = quantity

            if next_order_depth.sell_orders:
                visible_cross_volume = sum(
                    -vol for ask_price, vol in next_order_depth.sell_orders.items()
                    if ask_price <= price
                )
                if visible_cross_volume > 0:
                    filled = min(remaining, visible_cross_volume)
                    fills.append(
                        Trade(
                            symbol=symbol,
                            price=price,
                            quantity=filled,
                            buyer="SUBMISSION",
                            seller="",
                            timestamp=next_timestamp,
                        )
                    )
                    cash_delta -= price * filled
                    remaining -= filled

            improves_bid = (
                current_best_bid is not None
                and current_best_ask is not None
                and current_best_bid < price < current_best_ask
            )

            if remaining > 0 and improves_bid and next_total_trade_volume > 0:
                maker_fill = max(1, int(next_total_trade_volume * fill_fraction))
                maker_fill = min(remaining, maker_fill)

                fills.append(
                    Trade(
                        symbol=symbol,
                        price=price,
                        quantity=maker_fill,
                        buyer="SUBMISSION",
                        seller="",
                        timestamp=next_timestamp,
                    )
                )
                cash_delta -= price * maker_fill
                remaining -= maker_fill

        # PASSIVE SELL
        elif quantity < 0:
            remaining_to_sell = -quantity

            if next_order_depth.buy_orders:
                visible_cross_volume = sum(
                    vol for bid_price, vol in next_order_depth.buy_orders.items()
                    if bid_price >= price
                )
                if visible_cross_volume > 0:
                    filled = min(remaining_to_sell, visible_cross_volume)
                    fills.append(
                        Trade(
                            symbol=symbol,
                            price=price,
                            quantity=filled,
                            buyer="",
                            seller="SUBMISSION",
                            timestamp=next_timestamp,
                        )
                    )
                    cash_delta += price * filled
                    remaining_to_sell -= filled

            improves_ask = (
                current_best_bid is not None
                and current_best_ask is not None
                and current_best_bid < price < current_best_ask
            )

            if remaining_to_sell > 0 and improves_ask and next_total_trade_volume > 0:
                maker_fill = max(1, int(next_total_trade_volume * fill_fraction))
                maker_fill = min(remaining_to_sell, maker_fill)

                fills.append(
                    Trade(
                        symbol=symbol,
                        price=price,
                        quantity=maker_fill,
                        buyer="",
                        seller="SUBMISSION",
                        timestamp=next_timestamp,
                    )
                )
                cash_delta += price * maker_fill
                remaining_to_sell -= maker_fill

    return fills, cash_delta


def apply_fills_to_position(start_position: int, fills: List[Trade]) -> int:
    pos = start_position
    for trade in fills:
        if trade.buyer == "SUBMISSION":
            pos += trade.quantity
        elif trade.seller == "SUBMISSION":
            pos -= trade.quantity
    return pos


def get_mid(order_depth: OrderDepth):
    if order_depth.buy_orders and order_depth.sell_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    return None


def group_price_steps(prices: pd.DataFrame):
    """
    Baut eine Liste von Timesteps.
    Jeder Timestep enthält die Zeilen aller Produkte für (source_file, day, timestamp).
    """
    grouped_steps = []

    grouped = prices.groupby(["source_file", "day", "timestamp"], sort=True)
    for (source_file, day, timestamp), group in grouped:
        rows_by_product = {}
        for _, row in group.iterrows():
            rows_by_product[str(row["product"])] = row

        grouped_steps.append({
            "source_file": str(source_file),
            "day": int(day),
            "timestamp": int(timestamp),
            "rows_by_product": rows_by_product,
        })

    grouped_steps.sort(key=lambda x: (x["source_file"], x["day"], x["timestamp"]))
    return grouped_steps


def main():
    trader = Trader()

    prices = load_csvs(PRICE_FILES)
    trades = load_csvs(TRADE_FILES)

    prices = prices[prices["product"].isin(PRODUCTS)].copy()
    trades = trades[trades["symbol"].isin(PRODUCTS)].copy()

    steps = group_price_steps(prices)

    positions: Dict[str, int] = {p: 0 for p in PRODUCTS}
    cashs: Dict[str, int] = {p: 0 for p in PRODUCTS}
    trader_data = ""
    previous_own_trades: Dict[str, List[Trade]] = {p: [] for p in PRODUCTS}

    position_limits = getattr(trader, "POSITION_LIMITS", {})
    history_rows = []

    for i in range(len(steps)):
        step = steps[i]
        source_file = step["source_file"]
        day = step["day"]
        timestamp = step["timestamp"]
        rows_by_product = step["rows_by_product"]

        order_depths: Dict[str, OrderDepth] = {}
        market_trades: Dict[str, List[Trade]] = {}

        for product in PRODUCTS:
            if product in rows_by_product:
                order_depths[product] = build_order_depth(rows_by_product[product])
            else:
                order_depths[product] = OrderDepth()

            market_trades[product] = build_market_trades(trades, source_file, timestamp, product)

        state = TradingState(
            traderData=trader_data,
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades=previous_own_trades,
            market_trades=market_trades,
            position=positions.copy(),
            observations=Observation({}, {}),
        )

        result, conversions, trader_data = trader.run(state)

        current_step_own_trades: Dict[str, List[Trade]] = {p: [] for p in PRODUCTS}

        for product in PRODUCTS:
            product_orders = result.get(product, [])
            limit = position_limits.get(product, 20)
            current_position = positions.get(product, 0)

            product_orders = enforce_position_limits(product_orders, current_position, limit)

            immediate_fills: List[Trade] = []
            passive_fills: List[Trade] = []
            immediate_cash_delta = 0
            passive_cash_delta = 0
            residual_orders: List[Tuple[str, int, int]] = []

            if product in order_depths:
                immediate_fills, immediate_cash_delta, residual_orders = simulate_immediate_fills(
                    product,
                    product_orders,
                    order_depths[product],
                    timestamp,
                )

            if i < len(steps) - 1 and residual_orders:
                next_step = steps[i + 1]
                next_source_file = next_step["source_file"]
                next_timestamp = next_step["timestamp"]
                next_rows_by_product = next_step["rows_by_product"]

                next_order_depth = (
                    build_order_depth(next_rows_by_product[product])
                    if product in next_rows_by_product
                    else OrderDepth()
                )

                next_market_trades = build_market_trades(
                    trades,
                    next_source_file,
                    next_timestamp,
                    product,
                )

                passive_fills, passive_cash_delta = simulate_passive_fills(
                    residual_orders,
                    order_depths[product],
                    next_order_depth,
                    next_market_trades,
                    next_timestamp,
                )

            all_fills = immediate_fills + passive_fills
            total_cash_delta = immediate_cash_delta + passive_cash_delta

            cashs[product] += total_cash_delta
            positions[product] = apply_fills_to_position(positions[product], all_fills)
            current_step_own_trades[product] = all_fills

        previous_own_trades = current_step_own_trades

        # PnL-Snapshot pro Schritt
        emeralds_mid = get_mid(order_depths["EMERALDS"])
        tomatoes_mid = get_mid(order_depths["TOMATOES"])

        emeralds_mark = FAIR_VALUE_EMERALDS
        tomatoes_mark = tomatoes_mid if tomatoes_mid is not None else 0

        pnl_emeralds = cashs["EMERALDS"] + positions["EMERALDS"] * emeralds_mark
        pnl_tomatoes = cashs["TOMATOES"] + positions["TOMATOES"] * tomatoes_mark
        pnl_total = pnl_emeralds + pnl_tomatoes

        history_rows.append({
            "source_file": source_file,
            "day": day,
            "timestamp": timestamp,

            "emeralds_position": positions["EMERALDS"],
            "emeralds_cash": cashs["EMERALDS"],
            "emeralds_mid": emeralds_mid,
            "emeralds_pnl": pnl_emeralds,

            "tomatoes_position": positions["TOMATOES"],
            "tomatoes_cash": cashs["TOMATOES"],
            "tomatoes_mid": tomatoes_mid,
            "tomatoes_pnl": pnl_tomatoes,

            "total_pnl": pnl_total,
        })

    results = pd.DataFrame(history_rows)
    results.to_csv("backtest_results_all_products.csv", index=False)

    print("=" * 70)
    print("BACKTEST ABGESCHLOSSEN (EMERALDS + TOMATOES)")
    print("=" * 70)
    print(f"Anzahl Steps: {len(results)}")
    print()
    print("EMERALDS")
    print(f"  Finale Position: {positions['EMERALDS']}")
    print(f"  Final Cash: {cashs['EMERALDS']}")
    print(f"  Final PnL (Fair 10000): {cashs['EMERALDS'] + positions['EMERALDS'] * FAIR_VALUE_EMERALDS}")
    print()
    print("TOMATOES")
    print(f"  Finale Position: {positions['TOMATOES']}")
    tomatoes_final_mid = results.iloc[-1]['tomatoes_mid'] if len(results) > 0 else 0
    tomatoes_final_mark = tomatoes_final_mid if pd.notna(tomatoes_final_mid) else 0
    print(f"  Final Cash: {cashs['TOMATOES']}")
    print(f"  Final PnL (gegen letzten Mid): {cashs['TOMATOES'] + positions['TOMATOES'] * tomatoes_final_mark}")
    print()
    print("GESAMT")
    print(f"  Final Total PnL: {results.iloc[-1]['total_pnl'] if len(results) > 0 else 0}")
    print(f"  Best Total PnL: {results['total_pnl'].max() if len(results) > 0 else 0}")
    print(f"  Worst Total PnL: {results['total_pnl'].min() if len(results) > 0 else 0}")
    print(f"  Average Total PnL: {results['total_pnl'].mean():.2f if len(results) > 0 else 0}")
    print()
    print("Datei geschrieben: backtest_results_all_products.csv")
    print("Erste 10 Zeilen:")
    print(results.head(10))


if __name__ == "__main__":
    main()