from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import statistics
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


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
            trader_data = self.serialize_state()
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data

        order_depth: OrderDepth = state.order_depths["KELP"]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            trader_data = self.serialize_state()
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data

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
                logger.print(f"Golden cross detected. Buying {buy_qty} @ {best_ask}")

            # Death cross: Sell signal
            elif prev_short_ma > prev_long_ma and short_ma < long_ma and short_capacity > 0:
                sell_qty = min(self.base_quantity, short_capacity)
                orders.append(Order("KELP", best_bid, -sell_qty))
                logger.print(f"Death cross detected. Selling {sell_qty} @ {best_bid}")

        if orders:
            result["KELP"] = orders

        trader_data = self.serialize_state()
        logger.flush(state, result, conversions, trader_data)
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

            pass
