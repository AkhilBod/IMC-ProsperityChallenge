from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import statistics
import json

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.short_window = 100
        self.long_window = 300
        self.position_limit = 70
        self.base_quantity = 50

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        result = {}
        conversions = 0

        # Deserialize previous state
        self.deserialize_state(state.traderData)

        if "KELP" not in state.order_depths:
            return result, conversions, self.serialize_state()

        order_depth: OrderDepth = state.order_depths["KELP"]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result, conversions, self.serialize_state()

        # Compute mid price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update price history
        self.kelp_prices.append(mid_price)
        self.kelp_prices = self.kelp_prices[-(self.long_window + 2):]

        current_position = state.position.get("KELP", 0)
        long_capacity = self.position_limit - current_position
        short_capacity = self.position_limit + current_position

        orders: List[Order] = []

        # Ensure we have enough data for MAs
        if len(self.kelp_prices) >= self.long_window + 1:
            short_ma = statistics.mean(self.kelp_prices[-self.short_window:])
            long_ma = statistics.mean(self.kelp_prices[-self.long_window:])
            prev_short_ma = statistics.mean(self.kelp_prices[-self.short_window - 1:-1])
            prev_long_ma = statistics.mean(self.kelp_prices[-self.long_window - 1:-1])

            # Golden cross: Buy signal
            if prev_short_ma < prev_long_ma and short_ma > long_ma and long_capacity > 0:
                buy_qty = min(self.base_quantity, long_capacity)
                orders.append(Order("KELP", best_ask, buy_qty))

            # Death cross: Sell signal
            elif prev_short_ma > prev_long_ma and short_ma < long_ma and short_capacity > 0:
                sell_qty = min(self.base_quantity, short_capacity)
                orders.append(Order("KELP", best_bid, -sell_qty))

        if orders:
            result["KELP"] = orders

        trader_data = self.serialize_state()
        return result, conversions, trader_data

    def serialize_state(self) -> str:
        state = {
            "kelp_prices": self.kelp_prices
        }
        return json.dumps(state)

    def deserialize_state(self, state_str: str):
        if not state_str:
            return
        try:
            state = json.loads(state_str)
            self.kelp_prices = state.get("kelp_prices", self.kelp_prices)
        except json.JSONDecodeError:
            pass
