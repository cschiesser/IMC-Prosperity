from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,   
        "TOMATOES": 80,   
    }

    FAIR_VALUES = {
        "EMERALDS": 10000,
    }

    def bid(self):
        # nur für Round 1 relevant
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # traderData laden
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state, "EMERALDS")

        if "TOMATOES" in state.order_depths:
            tomato_orders, tomato_state = self.trade_tomatoes(state, "TOMATOES", data)
            result["TOMATOES"] = tomato_orders
            data["TOMATOES"] = tomato_state

        for product in state.order_depths:
            if product not in result:
                result[product] = []

        traderData = json.dumps(data)
        return result, conversions, traderData

    
    # EMERALDS
   
    def trade_emeralds(self, state: TradingState, product: str) -> List[Order]:
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]

        fair_value = self.FAIR_VALUES[product]
        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        temp_position = position

        # Aggressiv günstige Asks kaufen (strict!)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                ask_volume = -order_depth.sell_orders[ask_price]

                if ask_price < fair_value:
                    buy_capacity = limit - temp_position
                    if buy_capacity <= 0:
                        break

                    buy_size = min(ask_volume, buy_capacity)
                    if buy_size > 0:
                        orders.append(Order(product, ask_price, buy_size))
                        temp_position += buy_size
                else:
                    break

        # Aggressiv teure Bids verkaufen (strict!)
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                bid_volume = order_depth.buy_orders[bid_price]

                if bid_price > fair_value:
                    sell_capacity = limit + temp_position
                    if sell_capacity <= 0:
                        break

                    sell_size = min(bid_volume, sell_capacity)
                    if sell_size > 0:
                        orders.append(Order(product, bid_price, -sell_size))
                        temp_position -= sell_size
                else:
                    break

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        passive_bid = fair_value - 1
        passive_ask = fair_value + 1

        if best_bid is not None:
            passive_bid = min(fair_value - 1, best_bid + 1)

        if best_ask is not None:
            passive_ask = max(fair_value + 1, best_ask - 1)

        buy_capacity = limit - temp_position
        sell_capacity = limit + temp_position
        passive_size = 24

        if buy_capacity > 0 and temp_position < 44:
            buy_size = min(passive_size, buy_capacity)
            if buy_size > 0 and (best_ask is None or passive_bid < best_ask):
                orders.append(Order(product, passive_bid, buy_size))

        if sell_capacity > 0 and temp_position > -44:
            sell_size = min(passive_size, sell_capacity)
            if sell_size > 0 and (best_bid is None or passive_ask > best_bid):
                orders.append(Order(product, passive_ask, -sell_size))

        return orders


    # TOMATOES
  
    def trade_tomatoes(self, state: TradingState, product: str, data: Dict) -> tuple[List[Order], Dict]:
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]

        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        temp_position = position

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], data.get(product, {})

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2

        # Vorherigen Zustand laden
        product_state = data.get(product, {})
        previous_ema = product_state.get("ema_mid")

        # EMA für dynamischen Fair Value
        alpha = 0.2
        if previous_ema is None:
            ema_mid = current_mid
        else:
            ema_mid = alpha * current_mid + (1 - alpha) * previous_ema

        # Inventory-Skew
        inventory_skew_coeff = 0.3
        adjusted_fair = ema_mid - inventory_skew_coeff * position

        # Parameter für TOMATOES V1
        take_threshold = 1      # aggressiv nur bei echtem Edge
        quote_offset = 3        # passive Quotes um den Fair Value
        passive_size = 8        # konservativ starten

       
        # Aggressiv günstige Asks kaufen
        # wenn ask <= adjusted_fair - take_threshold
    
        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_volume = -order_depth.sell_orders[ask_price]

            if ask_price <= adjusted_fair - take_threshold:
                buy_capacity = limit - temp_position
                if buy_capacity <= 0:
                    break

                buy_size = min(ask_volume, buy_capacity)
                if buy_size > 0:
                    orders.append(Order(product, ask_price, buy_size))
                    temp_position += buy_size
            else:
                break

    
        # Aggressiv teure Bids verkaufen
        # wenn bid >= adjusted_fair + take_threshold

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_volume = order_depth.buy_orders[bid_price]

            if bid_price >= adjusted_fair + take_threshold:
                sell_capacity = limit + temp_position
                if sell_capacity <= 0:
                    break

                sell_size = min(bid_volume, sell_capacity)
                if sell_size > 0:
                    orders.append(Order(product, bid_price, -sell_size))
                    temp_position -= sell_size
            else:
                break


        #Passive Quotes um den dynamischen Fair Value
        
        passive_bid = math.floor(adjusted_fair) - quote_offset
        passive_ask = math.ceil(adjusted_fair) + quote_offset

        # etwas kompetitiver machen, ohne bewusst zu kreuzen
        passive_bid = max(passive_bid, best_bid + 1)
        passive_ask = min(passive_ask, best_ask - 1)

        buy_capacity = limit - temp_position
        sell_capacity = limit + temp_position

        # Wenn die Position schon stark long ist, weniger passiv kaufen
        if buy_capacity > 0 and temp_position < 30:
            buy_size = min(passive_size, buy_capacity)
            if buy_size > 0 and passive_bid < best_ask:
                orders.append(Order(product, passive_bid, buy_size))

        # Wenn die Position schon stark short ist, weniger passiv verkaufen
        if sell_capacity > 0 and temp_position > -30:
            sell_size = min(passive_size, sell_capacity)
            if sell_size > 0 and passive_ask > best_bid:
                orders.append(Order(product, passive_ask, -sell_size))

        new_state = {
            "ema_mid": ema_mid,
            "last_mid": current_mid,
            "adjusted_fair": adjusted_fair,
        }

        return orders, new_state