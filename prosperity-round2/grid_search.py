"""
Grid Search für trader2.py Parameter.
Lädt die Daten einmal, testet alle Kombinationen und gibt die Top-10 aus.
"""
import sys
import json
import math
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# backtest.py importiert "trader" — wir zeigen es auf trader2
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("trader", Path(__file__).parent / "trader2.py")
_mod  = _ilu.module_from_spec(_spec)
sys.modules["trader"] = _mod
_spec.loader.exec_module(_mod)

from backtest import (
    load_csvs, build_order_depth, build_market_trades,
    group_price_steps, simulate_immediate_fills, simulate_passive_fills,
    apply_fills_to_position, get_mid, enforce_position_limits,
    PRICE_FILES, TRADE_FILES,
)
from datamodel import OrderDepth, Trade, TradingState, Observation


PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]

# ──────────────────────────────────────────────
# Parameter-Raster — hier anpassen um mehr/weniger zu testen
# ──────────────────────────────────────────────
GRID = {
    # ASH_COATED_OSMIUM
    "ash_imbalance":    [3.0, 8.0],    # Gewicht des Order-Book-Signals
    "ash_take":         [0.5, 1.5],    # Take-Threshold
    # INTARIAN_PEPPER_ROOT
    "ema_alpha":        [0.08, 0.20],  # EMA-Glättung
    "target_scale":     [1.0, 1.3],   # Skaliert alle Zielinventare
    "sell_thresh":      [3.0, 6.0],   # Schwelle für aggressiven Sell
}


# ──────────────────────────────────────────────
# Konfigurierbarer Trader
# ──────────────────────────────────────────────
class TunableTrader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }
    ASH_BASE_FAIR = 10000

    def __init__(self, ash_imbalance, ash_take, ema_alpha, target_scale, sell_thresh):
        self.ash_imbalance  = ash_imbalance
        self.ash_take       = ash_take
        self.ema_alpha      = ema_alpha
        self.target_scale   = target_scale
        self.sell_thresh    = sell_thresh

    # ── Hilfsfunktionen ──────────────────────
    def _top(self, od: OrderDepth):
        bid = max(od.buy_orders)  if od.buy_orders  else None
        ask = min(od.sell_orders) if od.sell_orders else None
        return bid, ask

    def _mid(self, od: OrderDepth):
        bid, ask = self._top(od)
        if bid and ask: return (bid + ask) / 2
        return float(bid or ask or 0)

    def _imbalance(self, od: OrderDepth):
        bid, ask = self._top(od)
        if bid is None or ask is None: return 0.0
        bv = od.buy_orders.get(bid, 0)
        av = -od.sell_orders.get(ask, 0)
        d  = bv + av
        return (bv - av) / d if d > 0 else 0.0

    def _target(self, timestamp: int) -> int:
        raw = (
            55 if timestamp < 12000 else
            48 if timestamp < 30000 else
            40 if timestamp < 52000 else
            30 if timestamp < 76000 else
            18 if timestamp < 88000 else
            10 if timestamp < 94000 else
             4 if timestamp < 98000 else 0
        )
        return int(raw * self.target_scale)

    def _trend_rate(self, history: list) -> float:
        n = len(history)
        if n < 10: return 1.0 / 1000.0
        sx  = sum(p[0] for p in history)
        sy  = sum(p[1] for p in history)
        sxx = sum(p[0]**2 for p in history)
        sxy = sum(p[0]*p[1] for p in history)
        d   = n * sxx - sx**2
        return (n * sxy - sx * sy) / d if d != 0 else 1.0 / 1000.0

    # ── run ──────────────────────────────────
    def run(self, state: TradingState):
        result = {}
        data = {}
        if state.traderData:
            try: data = json.loads(state.traderData)
            except: pass

        lt = data.get("_last_timestamp")
        if lt is not None and state.timestamp < lt:
            data = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_ash(state)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            orders, ps = self._trade_ipr(state, data)
            result["INTARIAN_PEPPER_ROOT"] = orders
            data["IPR"] = ps

        for p in state.order_depths:
            if p not in result: result[p] = []

        data["_last_timestamp"] = state.timestamp
        return result, 0, json.dumps(data)

    # ── ASH_COATED_OSMIUM ────────────────────
    def _trade_ash(self, state: TradingState) -> list:
        orders = []
        od       = state.order_depths["ASH_COATED_OSMIUM"]
        limit    = self.POSITION_LIMITS["ASH_COATED_OSMIUM"]
        pos      = state.position.get("ASH_COATED_OSMIUM", 0)
        tmp      = pos
        bid, ask = self._top(od)
        imb      = self._imbalance(od)
        fv       = self.ASH_BASE_FAIR + self.ash_imbalance * imb - 0.08 * pos

        # Aggressive buys
        for ap in sorted(od.sell_orders):
            if ap > fv - self.ash_take: break
            cap = limit - tmp
            if cap <= 0: break
            sz = min(-od.sell_orders[ap], cap)
            if sz > 0:
                orders.append(type("O", (), {"symbol": "ASH_COATED_OSMIUM",
                                             "price": ap, "quantity": sz})())
                tmp += sz

        # Aggressive sells
        for bp in sorted(od.buy_orders, reverse=True):
            if bp < fv + self.ash_take: break
            cap = limit + tmp
            if cap <= 0: break
            sz = min(od.buy_orders[bp], cap)
            if sz > 0:
                orders.append(type("O", (), {"symbol": "ASH_COATED_OSMIUM",
                                             "price": bp, "quantity": -sz})())
                tmp -= sz

        # Passive
        pb = math.floor(fv) - 3
        pa = math.ceil(fv)  + 3
        if bid: pb = max(pb, bid + 1)
        if ask: pa = min(pa, ask - 1)

        if limit - tmp > 0 and tmp < 60:
            sz = min(30, limit - tmp)
            if ask is None or pb < ask:
                orders.append(type("O", (), {"symbol": "ASH_COATED_OSMIUM",
                                             "price": pb, "quantity": sz})())
        if limit + tmp > 0 and tmp > -60:
            sz = min(30, limit + tmp)
            if bid is None or pa > bid:
                orders.append(type("O", (), {"symbol": "ASH_COATED_OSMIUM",
                                             "price": pa, "quantity": -sz})())
        return orders

    # ── INTARIAN_PEPPER_ROOT ─────────────────
    def _trade_ipr(self, state: TradingState, data: dict):
        orders = []
        od    = state.order_depths["INTARIAN_PEPPER_ROOT"]
        limit = self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        pos   = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        tmp   = pos
        mid   = self._mid(od)
        if mid == 0: return [], data.get("IPR", {})

        bid, ask = self._top(od)
        imb      = self._imbalance(od)
        ps       = data.get("IPR", {})
        prev_ema = ps.get("base_ema")
        history  = ps.get("history", [])

        if ps.get("last_ts") is not None and state.timestamp < ps["last_ts"]:
            prev_ema = None
            history  = []

        history.append([state.timestamp, mid])
        if len(history) > 200: history = history[-200:]

        rate = self._trend_rate(history)
        det  = mid - rate * state.timestamp
        ema  = det if prev_ema is None else self.ema_alpha * det + (1 - self.ema_alpha) * prev_ema
        fair = ema + rate * state.timestamp

        tgt   = min(self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"], self._target(state.timestamp))
        gap   = pos - tgt
        adj   = fair + 4.0 * imb - 0.08 * gap

        late  = state.timestamp >= 93000
        final = state.timestamp >= 98000
        st    = self.sell_thresh
        if late:  st = max(st * 0.65, 2.5)
        if final: st = 1.0

        # Aggressive buys
        for ap in sorted(od.sell_orders):
            below = ap <= adj - 1.0
            catch = tmp < tgt and ap <= adj + 1.0
            if final: catch = tmp < tgt and ap <= adj
            if not (below or catch): break
            cap = limit - tmp
            if cap <= 0: break
            sz = min(-od.sell_orders[ap], cap)
            if sz > 0:
                orders.append(type("O", (), {"symbol": "INTARIAN_PEPPER_ROOT",
                                             "price": ap, "quantity": sz})())
                tmp += sz

        # Aggressive sells
        for bp in sorted(od.buy_orders, reverse=True):
            strong   = bp >= adj + st
            above    = tmp > tgt + 8
            unwind   = late  and tmp > tgt
            exit_fin = final and tmp > 0 and bp >= adj
            if not (tmp > 0 and ((strong and above) or unwind or exit_fin)): break
            cap = tmp
            if cap <= 0: break
            sz = min(od.buy_orders[bp], cap)
            if sz > 0:
                orders.append(type("O", (), {"symbol": "INTARIAN_PEPPER_ROOT",
                                             "price": bp, "quantity": -sz})())
                tmp -= sz

        # Passive
        pb = math.floor(adj) - 2
        pa = math.ceil(adj)  + 6
        if late:  pa = math.ceil(adj) + 4
        if bid: pb = max(pb, bid + 1)
        if ask: pa = min(pa, ask - 1)

        max_long = min(limit, tgt + 20)
        if late:  max_long = min(limit, tgt + 6)
        if final: max_long = tgt

        cap = limit - tmp
        if cap > 0 and tmp < max_long:
            sz = min(14, cap)
            if tmp < tgt - 16: sz = min(cap, 18)
            if ask is None or pb < ask:
                orders.append(type("O", (), {"symbol": "INTARIAN_PEPPER_ROOT",
                                             "price": pb, "quantity": sz})())

        allow_sell = tmp > max(tgt + 10, 12)
        if late  and tmp > tgt: allow_sell = True
        if final and tmp > 0:   allow_sell = True
        if allow_sell:
            cap = tmp
            if cap > 0:
                sz = min(14, cap)
                if tmp > tgt + 20: sz = min(cap, 18)
                if bid is None or pa > bid:
                    orders.append(type("O", (), {"symbol": "INTARIAN_PEPPER_ROOT",
                                                 "price": pa, "quantity": -sz})())

        return orders, {"base_ema": ema, "history": history,
                        "last_ts": state.timestamp, "rate": rate}


# ──────────────────────────────────────────────
# Backtest-Runner (einmaliges Laden der Daten)
# ──────────────────────────────────────────────
from datamodel import Order  # noqa: needed for Order objects below

# Patch: TunableTrader benutzt type("O",...) — wir brauchen echte Order-Objekte
# Überschreibe die _trade_* Methoden um echte Order-Objekte zu erzeugen.
# Einfachste Lösung: Order-Kompatibilität sicherstellen.
# datamodel.Order(symbol, price, quantity) — bereits importiert.

class _Ord:
    """Minimal Order-Wrapper kompatibel mit simulate_immediate_fills."""
    __slots__ = ("symbol", "price", "quantity")
    def __init__(self, symbol, price, quantity):
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity


def _patch_orders(orders, product):
    """Konvertiert generische Objekte zu _Ord falls nötig."""
    result = []
    for o in orders:
        result.append(_Ord(product, o.price, o.quantity))
    return result


def run_backtest(trader, steps, prices_df, trades_df) -> float:
    positions = {p: 0 for p in PRODUCTS}
    cashs     = {p: 0 for p in PRODUCTS}
    td        = ""
    prev_own  = {p: [] for p in PRODUCTS}
    limits    = getattr(trader, "POSITION_LIMITS", {})
    last_od   = {}

    for i, step in enumerate(steps):
        sf   = step["source_file"]
        ts   = step["timestamp"]
        rows = step["rows_by_product"]

        ods = {}
        mkt = {}
        for p in PRODUCTS:
            ods[p] = build_order_depth(rows[p]) if p in rows else OrderDepth()
            mkt[p] = build_market_trades(trades_df, sf, ts, p)
        last_od = ods

        state = TradingState(
            traderData=td, timestamp=ts, listings={},
            order_depths=ods, own_trades=prev_own,
            market_trades=mkt, position=positions.copy(),
            observations=Observation({}, {}),
        )

        result, _, td = trader.run(state)
        cur_own = {p: [] for p in PRODUCTS}

        for p in PRODUCTS:
            raw    = result.get(p, [])
            orders = _patch_orders(raw, p)
            limit  = limits.get(p, 20)
            orders = enforce_position_limits(orders, positions[p], limit)

            imm_fills, imm_cash, residuals = [], 0, []
            pas_fills, pas_cash            = [], 0

            if p in ods:
                imm_fills, imm_cash, residuals = simulate_immediate_fills(
                    p, orders, ods[p], ts)

            if i < len(steps) - 1 and residuals:
                nx  = steps[i + 1]
                nod = build_order_depth(nx["rows_by_product"][p]) if p in nx["rows_by_product"] else OrderDepth()
                nmt = build_market_trades(trades_df, nx["source_file"], nx["timestamp"], p)
                pas_fills, pas_cash = simulate_passive_fills(
                    residuals, ods[p], nod, nmt, nx["timestamp"])

            all_fills = imm_fills + pas_fills
            cashs[p]     += imm_cash + pas_cash
            positions[p]  = apply_fills_to_position(positions[p], all_fills)
            cur_own[p]    = all_fills

        prev_own = cur_own

    total = sum(
        cashs[p] + positions[p] * (get_mid(last_od[p]) or 0)
        for p in PRODUCTS
    )
    return total


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Lade Daten...")
    prices_df = load_csvs(PRICE_FILES)
    trades_df = load_csvs(TRADE_FILES)
    prices_df = prices_df[prices_df["product"].isin(PRODUCTS)].copy()
    trades_df = trades_df[trades_df["symbol"].isin(PRODUCTS)].copy()
    steps     = group_price_steps(prices_df)
    print(f"  {len(steps)} Timesteps geladen.\n")

    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f"Teste {total} Kombinationen...\n")

    results = []
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        trader = TunableTrader(
            ash_imbalance = params["ash_imbalance"],
            ash_take      = params["ash_take"],
            ema_alpha     = params["ema_alpha"],
            target_scale  = params["target_scale"],
            sell_thresh   = params["sell_thresh"],
        )
        pnl = run_backtest(trader, steps, prices_df, trades_df)
        results.append((pnl, params))

        if idx % 20 == 0 or idx == total:
            print(f"  [{idx}/{total}]  aktuell bestes PnL: "
                  f"{max(r[0] for r in results):,.0f}")

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER-KOMBINATIONEN")
    print("=" * 70)
    baseline = results[0][0]
    for rank, (pnl, p) in enumerate(results[:10], 1):
        print(f"\n#{rank}  PnL: {pnl:>10,.0f}")
        for k, v in p.items():
            print(f"      {k:<20} = {v}")

    print("\n" + "=" * 70)
    print("SCHLECHTESTE 3")
    print("=" * 70)
    for pnl, p in results[-3:]:
        print(f"  PnL: {pnl:>10,.0f}  |  {p}")

    # CSV speichern
    rows = []
    for pnl, p in results:
        row = {"pnl": pnl}
        row.update(p)
        rows.append(row)
    pd.DataFrame(rows).sort_values("pnl", ascending=False).to_csv(
        "grid_search_results.csv", index=False)
    print("\nAlle Ergebnisse gespeichert in: grid_search_results.csv")
