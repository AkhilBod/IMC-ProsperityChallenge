from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import statistics
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {
            product: [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.position_limit = 50
        self.max_order_size = 16
        self.products = ["KELP", "RESIN"]
        self.midprices = {product: [] for product in self.products}
        self.volatility = {product: 1 for product in self.products}
        self.ema_short = {product: None for product in self.products}
        self.ema_short_window = 10
        self.ema_long = {product: None for product in self.products}
        self.ema_long_window = 30

    def update_ema(self, price, current_ema, window):
        alpha = 2 / (window + 1)
        return price if current_ema is None else price * alpha + current_ema * (1 - alpha)

    def calculate_order_book_imbalance(self, order_depth):
        bid_vol = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
        ask_vol = abs(sum(order_depth.sell_orders.values())) if order_depth.sell_orders else 0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

    def calculate_fair_value(self, product, order_depth, market_trades, own_trades):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
        elif best_bid:
            mid_price = best_bid + 1
        elif best_ask:
            mid_price = best_ask - 1
        elif self.midprices[product]:
            mid_price = self.midprices[product][-1]
        else:
            mid_price = 2000 if product == "KELP" else 10000

        self.midprices[product].append(mid_price)
        if len(self.midprices[product]) > 10:
            self.midprices[product] = self.midprices[product][-10:]

        if len(self.midprices[product]) >= 2:
            self.volatility[product] = statistics.stdev(self.midprices[product])

        self.ema_short[product] = self.update_ema(mid_price, self.ema_short[product], self.ema_short_window)
        self.ema_long[product] = self.update_ema(mid_price, self.ema_long[product], self.ema_long_window)

        imbalance = self.calculate_order_book_imbalance(order_depth)
        fair_value = mid_price + imbalance * self.volatility[product] * 0.3

        if self.ema_short[product] is not None and self.ema_long[product] is not None:
            trend = self.ema_short[product] - self.ema_long[product]
            trend_adj = trend * (0.2 if product == "KELP" else 0.02)
            fair_value += trend_adj

        return fair_value

    def calculate_pricing(self, product, fair_value, order_depth, position):
        best_bid = max(order_depth.buy_orders.keys(), default=fair_value - 1)
        best_ask = min(order_depth.sell_orders.keys(), default=fair_value + 1)
        market_spread = best_ask - best_bid

        base_spread = max(1, market_spread // 2)
        vol_factor = max(1, min(3, self.volatility[product]))
        spread = max(1, base_spread * (1 + vol_factor * 0.2))

        position_factor = position / self.position_limit if self.position_limit else 0
        skew_adj = position_factor * spread * 0.5

        buy_price = round(fair_value - spread + skew_adj)
        sell_price = round(fair_value + spread + skew_adj)

        if buy_price >= sell_price:
            mid = (buy_price + sell_price) / 2
            buy_price = round(mid - 0.5)
            sell_price = round(mid + 0.5)

        remaining_buy = self.position_limit - position
        remaining_sell = self.position_limit + position
        buy_factor = min(1.0, remaining_buy / self.position_limit) if self.position_limit else 0
        sell_factor = min(1.0, remaining_sell / self.position_limit) if self.position_limit else 0
        vol_size_factor = max(0.5, 1 - (vol_factor - 1) / 4)

        buy_size = max(1, min(remaining_buy, round(self.max_order_size * buy_factor * vol_size_factor)))
        sell_size = max(1, min(remaining_sell, round(self.max_order_size * sell_factor * vol_size_factor)))

        return buy_price, sell_price, buy_size, sell_size

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        trader_data = ""

        if state.traderData:
            try:
                data = json.loads(state.traderData)
                self.midprices = data.get("midprices", self.midprices)
                self.ema_short = data.get("ema_short", self.ema_short)
                self.ema_long = data.get("ema_long", self.ema_long)
                self.volatility = data.get("volatility", self.volatility)
            except Exception as e:
                logger.print(f"Error parsing trader data: {e}")

        for product in self.products:
            if product not in state.order_depths:
                continue

            orders = []
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair_value = self.calculate_fair_value(product, order_depth, state.market_trades.get(product, []), state.own_trades.get(product, []))
            buy_price, sell_price, buy_size, sell_size = self.calculate_pricing(product, fair_value, order_depth, position)

            if position < self.position_limit and buy_size > 0:
                orders.append(Order(product, buy_price, buy_size))

            if -position < self.position_limit and sell_size > 0:
                orders.append(Order(product, sell_price, -sell_size))

            for price, volume in order_depth.sell_orders.items():
                if price < fair_value - self.volatility[product] and position < self.position_limit:
                    volume_to_buy = min(abs(volume), self.position_limit - position)
                    if volume_to_buy > 0:
                        orders.append(Order(product, price, volume_to_buy))
                        position += volume_to_buy

            for price, volume in order_depth.buy_orders.items():
                if price > fair_value + self.volatility[product] and -position < self.position_limit:
                    volume_to_sell = min(volume, self.position_limit + position)
                    if volume_to_sell > 0:
                        orders.append(Order(product, price, -volume_to_sell))
                        position -= volume_to_sell

            result[product] = orders

        trader_data = json.dumps({
            "midprices": {p: self.midprices[p][-10:] for p in self.products},
            "ema_short": self.ema_short,
            "ema_long": self.ema_long,
            "volatility": self.volatility
        })

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
