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

    # EMA smoothing factor
    EMA_ALPHA = 0.12

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
        if last_timestamp is not None and state.timestamp < last_timestamp:
            data = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.trade_ash_coated_osmium(
                state, "ASH_COATED_OSMIUM"
            )

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            intarian_orders, intarian_state = self.trade_intarian_pepper_root(
                state, "INTARIAN_PEPPER_ROOT", data
            )
            result["INTARIAN_PEPPER_ROOT"] = intarian_orders
            data["INTARIAN_PEPPER_ROOT"] = intarian_state

        for product in state.order_depths:
            if product not in result:
                result[product] = []

        data["_last_timestamp"] = state.timestamp
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

    def get_intarian_target_position(self, timestamp: int) -> int:
        if timestamp < 12000:
            return 70
        if timestamp < 30000:
            return 62
        if timestamp < 52000:
            return 52
        if timestamp < 76000:
            return 38
        if timestamp < 88000:
            return 20
        if timestamp < 94000:
            return 10
        if timestamp < 98000:
            return 4
        return 0

    # -------------------------
    # ASH_COATED_OSMIUM
    # -------------------------
    def trade_ash_coated_osmium(self, state: TradingState, product: str) -> List[Order]:
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]

        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        temp_position = position

        best_bid, best_ask = self.top_of_book(order_depth)
        imbalance = self.get_top_level_imbalance(order_depth)

        fair_value = self.ASH_BASE_FAIR + 5.0 * imbalance - 0.08 * position

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
    # INTARIAN_PEPPER_ROOT
    # -------------------------
    def trade_intarian_pepper_root(
        self, state: TradingState, product: str, data: Dict
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

        product_state = data.get(product, {})
        previous_base_ema = product_state.get("base_ema")
        previous_timestamp = product_state.get("last_timestamp")
        price_history = product_state.get("price_history", [])

        if previous_timestamp is not None and state.timestamp < previous_timestamp:
            previous_base_ema = None
            price_history = []

        price_history.append([state.timestamp, current_mid])
        if len(price_history) > 200:
            price_history = price_history[-200:]

        trend_rate = self._estimate_trend_rate(price_history)

        detrended_mid = current_mid - trend_rate * state.timestamp
        if previous_base_ema is None:
            base_ema = detrended_mid
        else:
            base_ema = self.EMA_ALPHA * detrended_mid + (1 - self.EMA_ALPHA) * previous_base_ema

        trend_fair = base_ema + trend_rate * state.timestamp

        target_position = self.get_intarian_target_position(state.timestamp)
        inventory_gap = position - target_position
        adjusted_fair = trend_fair + 4.0 * imbalance - 0.08 * inventory_gap

        min_position = 0
        late_unwind = state.timestamp >= 93000
        final_unwind = state.timestamp >= 98000

        buy_take_threshold = 1.0
        sell_take_threshold = 5.5
        passive_bid_offset = 1
        passive_ask_offset = 8
        passive_size = 20   # TUNED: was 14 → more volume per passive order

        if late_unwind:
            sell_take_threshold = 2.5
            passive_bid_offset = 3
            passive_ask_offset = 4
        if final_unwind:
            sell_take_threshold = 1.0
            passive_bid_offset = 4
            passive_ask_offset = 3

        # Aggressive Buys
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                ask_volume = -order_depth.sell_orders[ask_price]

                below_fair = ask_price <= adjusted_fair - buy_take_threshold
                catch_up_buy = (
                    temp_position < target_position and ask_price <= adjusted_fair + 2.5
                )
                should_buy = catch_up_buy or below_fair
                if final_unwind:
                    should_buy = temp_position < target_position and ask_price <= adjusted_fair

                if should_buy:
                    buy_capacity = limit - temp_position
                    if buy_capacity <= 0:
                        break
                    buy_size = min(ask_volume, buy_capacity)
                    if temp_position < target_position - 12:
                        buy_size = min(ask_volume, max(1, min(buy_capacity, ask_volume)))
                    if buy_size > 0:
                        orders.append(Order(product, ask_price, buy_size))
                        temp_position += buy_size
                else:
                    break

        # Aggressive Sells
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                bid_volume = order_depth.buy_orders[bid_price]

                strong_edge = bid_price >= adjusted_fair + sell_take_threshold
                very_above_target = temp_position > target_position + 8
                unwind_sell = late_unwind and temp_position > target_position
                final_exit = final_unwind and temp_position > 0 and bid_price >= adjusted_fair

                if temp_position > min_position and (
                    (strong_edge and very_above_target) or unwind_sell or final_exit
                ):
                    sell_capacity = temp_position - min_position
                    if sell_capacity <= 0:
                        break
                    sell_size = min(bid_volume, sell_capacity)
                    if sell_size > 0:
                        orders.append(Order(product, bid_price, -sell_size))
                        temp_position -= sell_size
                else:
                    break

        passive_bid = math.floor(adjusted_fair) - passive_bid_offset
        passive_ask = math.ceil(adjusted_fair) + passive_ask_offset

        if best_bid is not None:
            passive_bid = max(passive_bid, best_bid + 1)
        if best_ask is not None:
            passive_ask = min(passive_ask, best_ask - 1)

        buy_capacity = limit - temp_position
        max_long_for_passive_buy = min(limit, target_position + 20)
        if late_unwind:
            max_long_for_passive_buy = min(limit, target_position + 6)
        if final_unwind:
            max_long_for_passive_buy = target_position

        if buy_capacity > 0 and temp_position < max_long_for_passive_buy:
            buy_size = min(passive_size, buy_capacity)
            if temp_position < target_position - 16:
                buy_size = min(buy_capacity, passive_size + 4)
            if buy_size > 0 and (best_ask is None or passive_bid < best_ask):
                orders.append(Order(product, passive_bid, buy_size))

        allow_passive_sell = temp_position > max(target_position + 10, 12)
        if late_unwind and temp_position > target_position:
            allow_passive_sell = True
        if final_unwind and temp_position > 0:
            allow_passive_sell = True

        if allow_passive_sell:
            sell_capacity = temp_position - min_position
            if sell_capacity > 0:
                sell_size = min(passive_size, sell_capacity)
                if temp_position > target_position + 20:
                    sell_size = min(sell_capacity, passive_size + 4)
                if sell_size > 0 and (best_bid is None or passive_ask > best_bid):
                    orders.append(Order(product, passive_ask, -sell_size))

        new_state = {
            "base_ema": base_ema,
            "last_mid": current_mid,
            "trend_fair": trend_fair,
            "adjusted_fair": adjusted_fair,
            "target_position": target_position,
            "last_timestamp": state.timestamp,
            "price_history": price_history,
            "trend_rate": trend_rate,
        }

        return orders, new_state

    def _estimate_trend_rate(self, price_history: List) -> float:
        n = len(price_history)
        if n < 10:
            return 1.0 / 1000.0

        sum_x = sum(p[0] for p in price_history)
        sum_y = sum(p[1] for p in price_history)
        sum_xx = sum(p[0] ** 2 for p in price_history)
        sum_xy = sum(p[0] * p[1] for p in price_history)

        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            return 1.0 / 1000.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope
