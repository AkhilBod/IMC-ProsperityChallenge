from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
import statistics
import jsonpickle
import numpy as np
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
        # Enhanced data storage with more metrics
        self.traderData = {
            "prices": [],
            "spreads": [],
            "volumes": [],
            "position": 0,
            "last_timestamp": 0,
            "trend": 0,  # -1 for downtrend, 0 for neutral, 1 for uptrend
            "volatility": 0,
            "recent_trades": [],
            "last_fair_value": 2000.48  # Initialize with default value
        }
        
        # Optimized parameters
        self.params = {
            "position_limit": 50,
            "base_spread": 0.30,
            "avg_mid": 2000.00,
            "dynamic_spread_multiplier": .01,
            "aggression_level": 0.3,  # Dynamically adjusted
            "max_order_size": 50,
            "min_profit_threshold": 1.5,
            "trend_threshold": 1.2,  # Price change % to confirm trend
            "panic_sell_threshold": -0.25,  # % change for emergency exit
            "volatility_window": 20,
            "liquidity_absorption_factor": 0.02  # How much liquidity to take
        }
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        
        if "KELP" in state.order_depths:
            # Update position and timestamp
            self.traderData["position"] = state.position.get("KELP", 0)
            self.traderData["last_timestamp"] = state.timestamp
            
            # Process KELP orders
            order_depth = state.order_depths["KELP"]
            orders = self.enhanced_kelp_strategy(order_depth, state.timestamp)
            result["KELP"] = orders
            
            # Store recent trades for analysis
            if "KELP" in state.market_trades:
                self.traderData["recent_trades"] = state.market_trades["KELP"][-5:]  # Last 5 trades
        
        # Return the processed trader data as a string using jsonpickle
        trader_data = jsonpickle.encode(self.traderData)
        
        # No conversions for KELP
        conversions = 0
        logger.flush(state, result, conversions, trader_data)

        # Return the expected tuple format: orders, conversions, trader_data
        return result, conversions, trader_data
    
    def enhanced_kelp_strategy(self, order_depth: OrderDepth, timestamp: int) -> List[Order]:
        orders = []
        current_position = self.traderData["position"]
        
        # Get market data
        buy_prices = list(order_depth.buy_orders.keys())
        sell_prices = list(order_depth.sell_orders.keys())
        
        if not buy_prices or not sell_prices:
            return orders
            
        best_bid = max(buy_prices)
        best_ask = min(sell_prices)
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Update price history and calculate metrics
        self.update_market_metrics(mid_price, spread, timestamp)
        
        # Calculate dynamic parameters
        fair_value = self.calculate_dynamic_fair_value()
        # Store the fair value for future reference
        self.traderData["last_fair_value"] = fair_value
        
        trend_strength = self.assess_market_trend()
        volatility = self.calculate_volatility()
        
        # Adjust strategy based on market conditions
        self.adjust_strategy_parameters(trend_strength, volatility)
        
        # Determine order prices with dynamic spread
        bid_price, ask_price = self.calculate_order_prices(fair_value, spread)
        
        # Calculate position-aware order sizes
        max_buy_size = min(
            int(self.params["max_order_size"] * self.params["aggression_level"]),
            self.params["position_limit"] - current_position
        )
        max_sell_size = min(
            int(self.params["max_order_size"] * self.params["aggression_level"]),
            self.params["position_limit"] + current_position
        )
        
        # Fix: Ensure we don't exceed position limits
        max_buy_size = max(0, max_buy_size)
        max_sell_size = max(0, max_sell_size)
        
        # Generate orders based on enhanced logic
        aggressive_orders = self.generate_aggressive_orders(
            best_bid, best_ask, 
            fair_value, 
            max_buy_size, max_sell_size,
            trend_strength,
            order_depth  # Pass order depth for volume analysis
        )
        
        orders.extend(aggressive_orders)
        
        # Emergency exit check
        if self.needs_emergency_exit(mid_price, current_position):
            # Clear existing orders and add emergency exit order
            orders = []
            if current_position > 0:
                emergency_size = current_position
                orders.append(Order("KELP", best_bid, -emergency_size))
            elif current_position < 0:
                emergency_size = -current_position
                orders.append(Order("KELP", best_ask, emergency_size))
            return orders
        
        # Additional liquidity provision
        if spread > self.params["base_spread"] * 1.8 and not aggressive_orders:
            liquidity_orders = self.provide_liquidity(
                bid_price, ask_price,
                max_buy_size, max_sell_size
            )
            orders.extend(liquidity_orders)
        
        return orders
    
    def update_market_metrics(self, mid_price: float, spread: float, timestamp: int):
        """Update all market metrics and detect patterns"""
        self.traderData["prices"].append(mid_price)
        self.traderData["spreads"].append(spread)
        
        # Keep only recent data for responsiveness
        if len(self.traderData["prices"]) > 100:
            self.traderData["prices"] = self.traderData["prices"][-50:]
            self.traderData["spreads"] = self.traderData["spreads"][-50:]
    
    def calculate_dynamic_fair_value(self) -> float:
        """Calculate fair value using multiple methods"""
        if len(self.traderData["prices"]) < 5:
            return self.params["avg_mid"]
            
        # Weighted average of different methods
        try:
            simple_ma = statistics.mean(self.traderData["prices"][-10:])
            
            # Use numpy's weighted average with proper weights
            weights = list(range(1, min(21, len(self.traderData["prices"]) + 1)))
            price_window = self.traderData["prices"][-len(weights):]
            
            weighted_ma = np.average(price_window, weights=weights)
            
            # More weight to recent prices
            return (weighted_ma * 0.7) + (simple_ma * 0.3)
        except Exception:
            # Fallback to last fair value if calculation fails
            return self.traderData["last_fair_value"]
    
    def assess_market_trend(self) -> float:
        """Quantify market trend strength from -1 (strong down) to 1 (strong up)"""
        if len(self.traderData["prices"]) < 5:
            return 0
        
        try:    
            short_term = np.polyfit(
                range(min(5, len(self.traderData["prices"]))), 
                self.traderData["prices"][-min(5, len(self.traderData["prices"])):], 
                1
            )[0]
            
            if len(self.traderData["prices"]) >= 10:
                medium_term = np.polyfit(
                    range(min(10, len(self.traderData["prices"]))), 
                    self.traderData["prices"][-min(10, len(self.traderData["prices"])):], 
                    1
                )[0]
                return np.clip((short_term * 0.7 + medium_term * 0.3) * 10, -1, 1)  # Scaled and clamped
            return np.clip(short_term * 10, -1, 1)  # Clamped to [-1, 1]
        except Exception:
            return 0  # Safe fallback
    
    def calculate_volatility(self) -> float:
        """Calculate recent price volatility"""
        if len(self.traderData["prices"]) < 2:
            return 0
        
        try:
            window_size = min(self.params["volatility_window"], len(self.traderData["prices"]))
            return statistics.stdev(self.traderData["prices"][-window_size:])
        except Exception:
            return 0  # Safe fallback
    
    def adjust_strategy_parameters(self, trend_strength: float, volatility: float):
        """Dynamically adjust strategy based on market conditions"""
        # More aggressive in strong trends
        self.params["aggression_level"] = 1.0 + abs(trend_strength) * 0.5
        
        # Wider spreads in high volatility
        self.params["dynamic_spread_multiplier"] = 1.0 + (volatility / 10)
        
        # Reduce profit thresholds in strong trends
        if abs(trend_strength) > 0.5:
            self.params["min_profit_threshold"] = max(0.1, 0.5 - abs(trend_strength)/2)
        else:
            # Reset to default when trend is weak
            self.params["min_profit_threshold"] = 0.3
    
    def calculate_order_prices(self, fair_value: float, spread: float) -> tuple[float, float]:
        """Calculate bid/ask prices with dynamic adjustments"""
        base_half_spread = self.params["base_spread"] * 0.5
        spread_adjustment = max(base_half_spread, spread * 0.5 * self.params["dynamic_spread_multiplier"])
        
        # Round to nearest whole number
        bid_price = round(fair_value - spread_adjustment)
        ask_price = round(fair_value + spread_adjustment)
        
        return bid_price, ask_price
    
    def generate_aggressive_orders(self, best_bid: float, best_ask: float,
                                 fair_value: float, 
                                 max_buy: int, max_sell: int,
                                 trend_strength: float,
                                 order_depth: OrderDepth) -> List[Order]:
        """Generate orders with trend-adaptive logic"""
        orders = []
        
        # Calculate available volumes at best prices
        bid_volume = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
        
        # More aggressive in the direction of the trend
        if trend_strength < -0.3:  # Downtrend
            # Sell more aggressively
            sell_threshold = fair_value + self.params["min_profit_threshold"] * 0.7
            if max_sell > 0 and best_bid > sell_threshold:
                size = min(max_sell, int(bid_volume * self.params["liquidity_absorption_factor"]))
                if size > 0:
                    orders.append(Order("KELP", best_bid, -size))
                
        elif trend_strength > 0.3:  # Uptrend
            # Buy more aggressively
            buy_threshold = fair_value - self.params["min_profit_threshold"] * 0.7
            if max_buy > 0 and best_ask < buy_threshold:
                size = min(max_buy, int(ask_volume * self.params["liquidity_absorption_factor"]))
                if size > 0:
                    orders.append(Order("KELP", best_ask, size))
                
        else:  # Neutral market
            # Standard market making
            buy_threshold = fair_value - self.params["min_profit_threshold"]
            sell_threshold = fair_value + self.params["min_profit_threshold"]
            
            if max_buy > 0 and best_ask < buy_threshold:
                size = min(max_buy, int(ask_volume * 0.5))
                if size > 0:
                    orders.append(Order("KELP", best_ask, size))
                
            if max_sell > 0 and best_bid > sell_threshold:
                size = min(max_sell, int(bid_volume * 0.5))
                if size > 0:
                    orders.append(Order("KELP", best_bid, -size))
                
        return orders
    
    def needs_emergency_exit(self, current_price: float, position: int) -> bool:
        """Check if emergency exit conditions are met"""
        if len(self.traderData["prices"]) < 10 or position == 0:
            return False
            
        try:
            # Calculate recent price drop
            highest_recent = max(self.traderData["prices"][-10:])
            price_change = (current_price - highest_recent) / highest_recent
            
            # Emergency exit if:
            # 1) We're long and prices are dropping fast
            # 2) We're short and prices are rising fast
            if (position > 0 and price_change < self.params["panic_sell_threshold"]) or \
               (position < 0 and price_change > -self.params["panic_sell_threshold"]):
                return True
                
            return False
        except Exception:
            return False  # Default to safe behavior
    
    def provide_liquidity(self, bid_price: float, ask_price: float,
                         max_buy: int, max_sell: int) -> List[Order]:
        """Provide liquidity when spreads are wide"""
        orders = []
        if max_buy > 0:
            size = min(max_buy // 2, 5)  # Smaller orders for liquidity
            if size > 0:
                orders.append(Order("KELP", bid_price, size))
                
        if max_sell > 0:
            size = min(max_sell // 2, 5)
            if size > 0:
                orders.append(Order("KELP", ask_price, -size))
                
        return orders