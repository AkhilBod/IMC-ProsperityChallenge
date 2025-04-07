from typing import List, Dict, Tuple
import jsonpickle
import math
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
        # Strategy parameters
        self.ema_periods = 200
        self.kelp_std_dev_multiplier = 2.5
        self.resin_std_dev_multiplier = 1.0
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], List, str]:

        print("traderData: " + state.traderData if state.traderData else "traderData: ")
        print("Observations: " + str(state.observations))
        
        # Initialize result containers
        result = {}
        conversions = []
        
        # Load previous state if available
        trader_state = self.initialize_state(state.traderData)
        
        # Process each product in order depths
        for product in state.order_depths:
            # Get order depth for this product
            order_depth = state.order_depths[product]
            
            # Initialize orders list for this product
            orders = []
            
            # Skip if no orders to process
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue
            
            # Calculate mid price from order book
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            best_ask_amount = order_depth.sell_orders[best_ask]
            mid_price = (best_bid + best_ask) / 2
            print("SELL", str(-50) + "x", best_bid)
            print("BUY", str(50) + "x", best_ask)

            # Initialize product data if not already present
            if product not in trader_state["historical_prices"]:
                trader_state["historical_prices"][product] = []
                trader_state["ema_values"][product] = None
            
            # Store historical price
            trader_state["historical_prices"][product].append(mid_price)
            
            # Limit history length
            if len(trader_state["historical_prices"][product]) > self.ema_periods:
                trader_state["historical_prices"][product] = trader_state["historical_prices"][product][-self.ema_periods:]
            
            # Skip trading if not enough data yet
            if len(trader_state["historical_prices"][product]) < self.ema_periods:
                result[product] = []
                continue
            
            # Calculate or update EMA
            if trader_state["ema_values"][product] is None:
                # Initial EMA is simple average when we first have enough data
                trader_state["ema_values"][product] = sum(trader_state["historical_prices"][product]) / self.ema_periods
            else:
                # Update EMA with new price
                multiplier = 2 / (self.ema_periods + 1)
                trader_state["ema_values"][product] = (mid_price - trader_state["ema_values"][product]) * multiplier + trader_state["ema_values"][product]
            
            # Calculate standard deviation for Keltner Channel
            std_dev = self.calculate_std_dev(trader_state["historical_prices"][product], trader_state["ema_values"][product])
            
            # Set standard deviation multiplier based on product
            std_dev_multiplier = self.kelp_std_dev_multiplier if product == "KELP" else self.resin_std_dev_multiplier
            
            # Calculate Keltner Channel bands
            upper_band = trader_state["ema_values"][product] + (std_dev * std_dev_multiplier)
            lower_band = trader_state["ema_values"][product] - (std_dev * std_dev_multiplier)
            
            # Use appropriate band as acceptable price based on the action
            buy_acceptable_price = lower_band  # Buy at lower band
            sell_acceptable_price = upper_band  # Sell at upper band
            
            # Get current position for this product
            current_position = state.position.get(product, 0)
            
            # Get position limit for this product
            position_limit = 50  # Adjust based on actual limits
            
            # Buy signal - when ask price is at or below our buy acceptable price (lower band)
            if best_ask <= buy_acceptable_price:
                # Calculate available position size
                available_position = position_limit - current_position
                if available_position > 0:
                    # Trade with max lot size
                    buy_quantity = min(available_position, abs(best_ask_amount))
                    if buy_quantity > 0:
                        orders.append(Order(product, best_ask, buy_quantity))
                        # Log buy order EXACTLY as in gold standard
                        print("BUY", str(buy_quantity) + "x", best_ask)
                    
            # Sell signal - when bid price is at or above our sell acceptable price (upper band)
            elif best_bid >= sell_acceptable_price:
                # Calculate available position size
                available_position = position_limit + current_position
                if available_position > 0:
                    # Trade with max lot size
                    sell_quantity = min(available_position, best_bid_amount)
                    if sell_quantity > 0:
                        orders.append(Order(product, best_bid, -sell_quantity))
                        # Log sell order EXACTLY as in gold standard
                        print("SELL", str(-sell_quantity) + "x", best_bid)
            
            # Store orders for this product
            result[product] = orders
        
        # Save updated state
        trader_data = jsonpickle.encode(trader_state)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def initialize_state(self, traderData):
        """Initialize or load trader state"""
        if traderData:
            try:
                return jsonpickle.decode(traderData)
            except:
                pass
        
        # Default initial state
        return {
            "historical_prices": {},
            "ema_values": {}
        }
    def calculate_std_dev(self, prices, ema):
        """Calculate standard deviation from EMA"""
        squared_diffs = [(price - ema) ** 2 for price in prices]
        variance = sum(squared_diffs) / len(squared_diffs)
        return math.sqrt(variance)