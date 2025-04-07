from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any, Tuple
import statistics
import json
import jsonpickle
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "", ""
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [d.buy_orders, d.sell_orders] for s, d in depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for ts in trades.values() for t in ts]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {
            p: [
                o.bidPrice, o.askPrice, o.transportFees,
                o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for os in orders.values() for o in os]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.short_window = 100
        self.long_window = 300
        self.kelp_prices = []
        self.kelp_position_limit = 70
        self.kelp_base_quantity = 50

        self.ema_periods = 200
        self.kelp_std_dev_multiplier = 2.5
        self.resin_std_dev_multiplier = 1.0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = self.initialize_state(state.traderData)

        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            mid_price = (best_bid + best_ask) / 2
            current_position = state.position.get(product, 0)

            if product == "KELP":
                self.kelp_prices.append(mid_price)
                self.kelp_prices = self.kelp_prices[-(self.long_window + 2):]
                long_capacity = self.kelp_position_limit - current_position
                short_capacity = self.kelp_position_limit + current_position
                orders = []

                if len(self.kelp_prices) >= self.long_window + 1:
                    short_ma = statistics.mean(self.kelp_prices[-self.short_window:])
                    long_ma = statistics.mean(self.kelp_prices[-self.long_window:])
                    prev_short_ma = statistics.mean(self.kelp_prices[-self.short_window - 1:-1])
                    prev_long_ma = statistics.mean(self.kelp_prices[-self.long_window - 1:-1])

                    if prev_short_ma < prev_long_ma and short_ma > long_ma and long_capacity > 0:
                        qty = min(self.kelp_base_quantity, long_capacity)
                        orders.append(Order(product, best_ask, qty))
                        logger.print(f"Golden cross detected. Buying {qty} @ {best_ask}")

                    elif prev_short_ma > prev_long_ma and short_ma < long_ma and short_capacity > 0:
                        qty = min(self.kelp_base_quantity, short_capacity)
                        orders.append(Order(product, best_bid, -qty))
                        logger.print(f"Death cross detected. Selling {qty} @ {best_bid}")

                result[product] = orders
            else:
                # EMA + Keltner logic
                if product not in trader_state["historical_prices"]:
                    trader_state["historical_prices"][product] = []
                    trader_state["ema_values"][product] = None

                trader_state["historical_prices"][product].append(mid_price)
                trader_state["historical_prices"][product] = trader_state["historical_prices"][product][-self.ema_periods:]

                if len(trader_state["historical_prices"][product]) < self.ema_periods:
                    result[product] = []
                    continue

                if trader_state["ema_values"][product] is None:
                    trader_state["ema_values"][product] = sum(trader_state["historical_prices"][product]) / self.ema_periods
                else:
                    multiplier = 2 / (self.ema_periods + 1)
                    trader_state["ema_values"][product] = (mid_price - trader_state["ema_values"][product]) * multiplier + trader_state["ema_values"][product]

                std_dev = self.calculate_std_dev(trader_state["historical_prices"][product], trader_state["ema_values"][product])
                std_multiplier = self.resin_std_dev_multiplier
                upper_band = trader_state["ema_values"][product] + std_multiplier * std_dev
                lower_band = trader_state["ema_values"][product] - std_multiplier * std_dev

                position_limit = 50
                orders = []

                if best_ask < lower_band:
                    available_position = position_limit - current_position
                    qty = min(available_position, abs(best_ask_amount))
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        logger.print("BUY", f"{qty}x @ {best_ask}")

                elif best_bid > upper_band:
                    available_position = position_limit + current_position
                    qty = min(available_position, best_bid_amount)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        logger.print("SELL", f"{qty}x @ {best_bid}")

                result[product] = orders

        trader_data = self.serialize_state(trader_state)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def serialize_state(self, state) -> str:
        return jsonpickle.encode({"historical_prices": state["historical_prices"], "ema_values": state["ema_values"], "kelp_prices": self.kelp_prices})

    def initialize_state(self, traderData):
        if traderData:
            try:
                data = jsonpickle.decode(traderData)
                self.kelp_prices = data.get("kelp_prices", [])
                return {
                    "historical_prices": data.get("historical_prices", {}),
                    "ema_values": data.get("ema_values", {})
                }
            except Exception:
                pass
        return {"historical_prices": {}, "ema_values": {}}

    def calculate_std_dev(self, prices, ema):
        squared_diffs = [(p - ema) ** 2 for p in prices]
        return math.sqrt(sum(squared_diffs) / len(squared_diffs))
