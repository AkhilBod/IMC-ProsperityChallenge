from typing import List, Dict, Tuple
import jsonpickle
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
        self.buy_price = 9998
        self.sell_price = 10002
        self.mid_price = 10000
        self.position_limit = 50
        self.take_profit_percentage = 0  # 50% of position at mid price
        
        # Exit prices for max positions
        self.long_exit_price = 10000  # Exit price when at max long position
        self.short_exit_price = 10000  # Exit price when at max short position
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = self.initialize_state(state.traderData)
        
        # We focus only on RAINFOREST_RESIN
        product = "RAINFOREST_RESIN"
        
        # Skip if product not in order depths
        if product not in state.order_depths:
            result[product] = []
            trader_data = jsonpickle.encode(trader_state)
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data
        
        # Get order depth for this product
        order_depth = state.order_depths[product]
        
        # Initialize orders list for this product
        orders = []
        
        # Get current position
        current_position = state.position.get(product, 0)
        
        # Check if we're at max position
        at_max_long = current_position >= self.position_limit
        at_max_short = current_position <= -self.position_limit
        
        # Check if our exit prices are available in the market
        long_exit_available = False
        short_exit_available = False
        
        for price, quantity in order_depth.buy_orders.items():
            if price >= self.long_exit_price:
                long_exit_available = True
                break
                
        for price, quantity in order_depth.sell_orders.items():
            if price <= self.short_exit_price:
                short_exit_available = True
                break
        
        # Handle max position exits
        if at_max_long and long_exit_available:
            # Exit the long position at our defined exit price
            orders.append(Order(product, self.long_exit_price, -self.position_limit))
            logger.print("EXIT LONG", str(self.position_limit) + "x", self.long_exit_price)
            result[product] = orders
            trader_data = jsonpickle.encode(trader_state)
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data
            
        if at_max_short and short_exit_available:
            # Exit the short position at our defined exit price
            orders.append(Order(product, self.short_exit_price, self.position_limit))
            logger.print("EXIT SHORT", str(self.position_limit) + "x", self.short_exit_price)
            result[product] = orders
            trader_data = jsonpickle.encode(trader_state)
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data
        
        # Normal trading logic if not at max positions or exit prices not available
        
        # Check if there are orders available at our target prices
        buy_available = False
        sell_available = False
        
        # Check if our buy price exists in the sell orders
        for price, quantity in order_depth.sell_orders.items():
            if price <= self.buy_price:
                buy_available = True
                break
                
        # Check if our sell price exists in the buy orders
        for price, quantity in order_depth.buy_orders.items():
            if price >= self.sell_price:
                sell_available = True
                break
        
        # Check if our mid price exists in the buy orders for taking profits
        take_profit_available = False
        for price, quantity in order_depth.buy_orders.items():
            if price >= self.mid_price:
                take_profit_available = True
                break
        
        # Buy when price reaches our buy target and we have room in position
        if buy_available and current_position < self.position_limit:
            # Calculate how much we can buy
            buy_quantity = self.position_limit - current_position
            if buy_quantity > 0:
                orders.append(Order(product, self.buy_price, buy_quantity))
                logger.print("BUY", str(buy_quantity) + "x", self.buy_price)
        
        # Sell when price reaches our sell target and we have inventory
        if sell_available and current_position > -self.position_limit:
            # Calculate how much we can sell
            sell_quantity = self.position_limit + current_position
            if sell_quantity > 0:
                orders.append(Order(product, self.sell_price, -sell_quantity))
                logger.print("SELL", str(sell_quantity) + "x", self.sell_price)
        
        # Take profit at mid price if we have a positive position
        if take_profit_available and current_position > 0:
            # Calculate how much to sell for profit
            profit_quantity = int(current_position * self.take_profit_percentage)
            if profit_quantity > 0:
                orders.append(Order(product, self.mid_price, -profit_quantity))
                logger.print("TAKE PROFIT", str(profit_quantity) + "x", self.mid_price)
        
        result[product] = orders
        trader_data = jsonpickle.encode(trader_state)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def initialize_state(self, traderData):
        if traderData:
            try:
                return jsonpickle.decode(traderData)
            except:
                pass
        
        return {}