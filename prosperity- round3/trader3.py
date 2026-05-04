from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ASH_BASE_FAIR = 10000

    # EMA smoothing factor for ASH (mean-reverting product)
    EMA_ALPHA = 0.12

    # PEPPER rises at ~0.001/timestamp (~1000/day), 3 days total (day -1, 0, 1)
    PEPPER_TREND = 0.001

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        last_timestamp = data.get("_last_timestamp")
        day_count = data.get("_day_count", 0)
        if last_timestamp is not None and state.timestamp < last_timestamp:
            # New day started — increment day counter, reset product states
            day_count += 1
            data = {"_day_count": day_count}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.trade_ash_coated_osmium(
                state, "ASH_COATED_OSMIUM", data
            )

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            pepper_orders, pepper_state = self.trade_pepper(
                state, "INTARIAN_PEPPER_ROOT", data, day_count
            )
            result["INTARIAN_PEPPER_ROOT"] = pepper_orders
            data["INTARIAN_PEPPER_ROOT"] = pepper_state

        for product in state.order_depths:
            if product not in result:
                result[product] = []

        data["_last_timestamp"] = state.timestamp
        data["_day_count"] = day_count
        traderData = json.dumps(data)
        return result, conversions, traderData

    def top_of_book(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.top_of_book(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def get_top_level_imbalance(self, order_depth: OrderDepth) -> float:
        best_bid, best_ask = self.top_of_book(order_depth)
        if best_bid is None or best_ask is None:
            return 0.0
        bid_vol = order_depth.buy_orders.get(best_bid, 0)
        ask_vol = -order_depth.sell_orders.get(best_ask, 0)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def get_pepper_target_position(self, day_count: int, timestamp: int) -> int:
        """
        Days -1 and 0 (day_count 0, 1): hold max long — price rises all 3 days.
        Day 1 (day_count 2): hold until ts=940000 then fast unwind.
        """
        if day_count < 2:
            return 80
        # Last day: hold as long as possible, then fast close
        if timestamp < 940000:
            return 80
        if timestamp < 960000:
            return 50
        if timestamp < 975000:
            return 20
        if timestamp < 988000:
            return 4
        return 0

    # -------------------------
    # ASH_COATED_OSMIUM — mean-reverting market maker around 10000
    # -------------------------
    def trade_ash_coated_osmium(self, state: TradingState, product: str, data: Dict) -> List[Order]:
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]

        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        temp_position = position

        best_bid, best_ask = self.top_of_book(order_depth)
        imbalance = self.get_top_level_imbalance(order_depth)

        # EMA of mid to track slow drift in ASH fair value
        current_mid = self.get_mid_price(order_depth)
        ash_state = data.get(product, {})
        prev_ema = ash_state.get("ema")
        if current_mid is not None:
            if prev_ema is None:
                ema = current_mid
            else:
                ema = self.EMA_ALPHA * current_mid + (1 - self.EMA_ALPHA) * prev_ema
        else:
            ema = self.ASH_BASE_FAIR

        fair_value = ema + 2.0 * imbalance - 0.02 * position
        data[product] = {"ema": ema}

        take_threshold = 0.5
        passive_offset = 3
        passive_size = 30

        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                ask_volume = -order_depth.sell_orders[ask_price]
                if ask_price <= fair_value - take_threshold:
                    buy_capacity = limit - temp_position
                    if buy_capacity <= 0:
                        break
                    buy_size = min(ask_volume, buy_capacity)
                    if buy_size > 0:
                        orders.append(Order(product, ask_price, buy_size))
                        temp_position += buy_size
                else:
                    break

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                bid_volume = order_depth.buy_orders[bid_price]
                if bid_price >= fair_value + take_threshold:
                    sell_capacity = limit + temp_position
                    if sell_capacity <= 0:
                        break
                    sell_size = min(bid_volume, sell_capacity)
                    if sell_size > 0:
                        orders.append(Order(product, bid_price, -sell_size))
                        temp_position -= sell_size
                else:
                    break

        passive_bid = math.floor(fair_value) - passive_offset
        passive_ask = math.ceil(fair_value) + passive_offset

        if best_bid is not None:
            passive_bid = max(passive_bid, best_bid + 1)
        if best_ask is not None:
            passive_ask = min(passive_ask, best_ask - 1)

        buy_capacity = limit - temp_position
        sell_capacity = limit + temp_position

        if buy_capacity > 0 and temp_position < 60:
            buy_size = min(passive_size, buy_capacity)
            if buy_size > 0 and (best_ask is None or passive_bid < best_ask):
                orders.append(Order(product, passive_bid, buy_size))

        if sell_capacity > 0 and temp_position > -60:
            sell_size = min(passive_size, sell_capacity)
            if sell_size > 0 and (best_bid is None or passive_ask > best_bid):
                orders.append(Order(product, passive_ask, -sell_size))

        return orders

    # -------------------------
    # INTARIAN_PEPPER_ROOT — linear trend buy-and-hold
    # -------------------------
    def trade_pepper(
        self, state: TradingState, product: str, data: Dict, day_count: int
    ) -> Tuple[List[Order], Dict]:
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]

        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        temp_position = position

        current_mid = self.get_mid_price(order_depth)
        if current_mid is None:
            return [], data.get(product, {})

        best_bid, best_ask = self.top_of_book(order_depth)
        imbalance = self.get_top_level_imbalance(order_depth)

        # Estimate fair value using EMA of detrended price
        product_state = data.get(product, {})
        price_history = product_state.get("price_history", [])
        prev_base_ema = product_state.get("base_ema")

        price_history.append([state.timestamp, current_mid])
        if len(price_history) > 300:
            price_history = price_history[-300:]

        trend_rate = self._estimate_trend_rate(price_history)

        detrended = current_mid - trend_rate * state.timestamp
        if prev_base_ema is None:
            base_ema = detrended
        else:
            base_ema = self.EMA_ALPHA * detrended + (1 - self.EMA_ALPHA) * prev_base_ema

        fair_value = base_ema + trend_rate * state.timestamp + 3.0 * imbalance

        target_position = self.get_pepper_target_position(day_count, state.timestamp)
        inventory_gap = position - target_position

        is_unwinding = day_count >= 2 and state.timestamp >= 800000
        is_final_close = day_count >= 2 and state.timestamp >= 980000

        # Thresholds
        buy_take_threshold = 1.5
        sell_take_threshold = 1.5
        passive_bid_offset = 2
        passive_ask_offset = 2
        passive_size = 25

        if is_unwinding:
            buy_take_threshold = 99999  # no more buys during unwind
            sell_take_threshold = 1.0
            passive_bid_offset = 999
            passive_ask_offset = 1
        if is_final_close:
            sell_take_threshold = 0.0

        # Aggressive Buys — accumulate up to target
        if not is_unwinding and order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                ask_volume = -order_depth.sell_orders[ask_price]

                below_fair = ask_price <= fair_value - buy_take_threshold
                # More aggressive if far from target — price trend makes overpaying cheap
                catch_up_slack = 8.0 if temp_position < target_position * 0.7 else 4.0
                catch_up = temp_position < target_position and ask_price <= fair_value + catch_up_slack

                if below_fair or catch_up:
                    buy_capacity = limit - temp_position
                    if buy_capacity <= 0:
                        break
                    buy_size = min(ask_volume, buy_capacity)
                    if buy_size > 0:
                        orders.append(Order(product, ask_price, buy_size))
                        temp_position += buy_size
                else:
                    break

        # Aggressive Sells — only when unwinding or strong edge while over target
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                bid_volume = order_depth.buy_orders[bid_price]

                should_sell = False
                if is_unwinding and temp_position > target_position:
                    should_sell = bid_price >= fair_value - sell_take_threshold
                elif not is_unwinding and temp_position > target_position + 5:
                    should_sell = bid_price >= fair_value + sell_take_threshold

                if should_sell:
                    sell_capacity = temp_position - max(0, target_position)
                    if sell_capacity <= 0:
                        break
                    sell_size = min(bid_volume, sell_capacity)
                    if sell_size > 0:
                        orders.append(Order(product, bid_price, -sell_size))
                        temp_position -= sell_size
                else:
                    break

        # Passive orders
        passive_bid = math.floor(fair_value) - passive_bid_offset
        passive_ask = math.ceil(fair_value) + passive_ask_offset

        if best_bid is not None:
            passive_bid = max(passive_bid, best_bid + 1)
        if best_ask is not None:
            passive_ask = min(passive_ask, best_ask - 1)

        buy_capacity = limit - temp_position
        if not is_unwinding and buy_capacity > 0 and temp_position < target_position:
            buy_size = min(passive_size, buy_capacity)
            if buy_size > 0 and (best_ask is None or passive_bid < best_ask):
                orders.append(Order(product, passive_bid, buy_size))

        if is_unwinding and temp_position > target_position:
            sell_capacity = temp_position - target_position
            if sell_capacity > 0:
                sell_size = min(passive_size, sell_capacity)
                if sell_size > 0 and (best_bid is None or passive_ask > best_bid):
                    orders.append(Order(product, passive_ask, -sell_size))

        new_state = {
            "base_ema": base_ema,
            "price_history": price_history,
            "trend_rate": trend_rate,
            "fair_value": fair_value,
            "target_position": target_position,
        }

        return orders, new_state

    def _estimate_trend_rate(self, price_history: List) -> float:
        n = len(price_history)
        if n < 10:
            return self.PEPPER_TREND

        sum_x = sum(p[0] for p in price_history)
        sum_y = sum(p[1] for p in price_history)
        sum_xx = sum(p[0] ** 2 for p in price_history)
        sum_xy = sum(p[0] * p[1] for p in price_history)

        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            return self.PEPPER_TREND

        slope = (n * sum_xy - sum_x * sum_y) / denom
        # Clamp to plausible range
        return max(0.0005, min(0.002, slope))