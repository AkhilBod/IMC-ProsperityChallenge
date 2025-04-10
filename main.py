from typing import List, Dict, Tuple, Any
import jsonpickle
import json
import statistics
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, traderData: str) -> None:
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

        # We truncate state.traderData, traderData, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(traderData, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, traderData: str) -> list[Any]:
        return [
            state.timestamp,
            traderData,
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

class Product:
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    SQUID = "SQUID_INK"

class Trader:
    def __init__(self):
        # Shared data storage for all products
        self.trader_data = {
            Product.KELP: {
                "prices": [],
                "spreads": [],
                "volumes": [],
                "position": 0,
                "last_timestamp": 0,
                "trend": 0,
                "volatility": 0,
                "recent_trades": [],
                "last_fair_value": 2000.48
            },
            Product.RESIN: {
                "prices": [],
                "position": 0,
                "last_fair_value": 10000
            },
            Product.SQUID: {
                "prices": [],
                "spreads": [],
                "volumes": [],
                "position": 0,
                "last_timestamp": 0,
                "trend": 0,
                "volatility": 0,
                "recent_trades": [],
                "last_fair_value": 5000.0,
                "price_extremes": {"min": 4500.0, "max": 5500.0},
                "volatility_history": [],
                "trade_count": 0,
                "profit_loss": 0.0,
                "price_jumps": []
            }
        }
        
        # Product-specific parameters
        self.params = {
            Product.KELP: {
                "position_limit": 50,
                "base_spread": 0.15,
                "avg_mid": 2000.00,
                "dynamic_spread_multiplier": 0.005,
                "aggression_level": 0.3,
                "max_order_size": 20,
                "min_profit_threshold": 1.5,
                "trend_threshold": 1.2,
                "panic_sell_threshold": -0.25,
                "volatility_window": 20,
                "liquidity_absorption_factor": 0.02,
                # From first implementation
                "fair_value": 2000.0,
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 40
            },
            Product.RESIN: {
                "position_limit": 50,
                "fair_value": 10000,
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 40
            },
            Product.SQUID: {
                "position_limit": 20,
                "base_spread": 5.0,
                "default_fair_value": 5000.0,
                "dynamic_spread_multiplier": 0.1,
                "aggression_level": 0.3,
                "max_order_size": 5,
                "min_profit_threshold": 5.0,
                "trend_threshold": 2.0,
                "panic_sell_threshold": -1.5,
                "volatility_window": 15,
                "liquidity_absorption_factor": 0.1,
                "soft_position_limit": 15,
                "price_jump_threshold": 20.0,
                "mean_reversion_strength": 0.6,
                "volatility_bands_width": 2.0
            }
        }

    #########################
    # SHARED UTILITY METHODS
    #########################
    
    def calculate_volatility(self, prices, window=15):
        """Calculate recent price volatility"""
        if len(prices) < 2:
            return 0
        
        window_size = min(window, len(prices))
        try:
            return statistics.stdev(prices[-window_size:])
        except:
            return 0
    
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[int, int]:
        position_limit = self.params[product]["position_limit"]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        position_limit = self.params[product]["position_limit"]
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        position_limit = self.params[product]["position_limit"]
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    ###############################
    # KELP STRATEGY IMPLEMENTATION
    ###############################
    
    def calculate_kelp_fair_value(self):
        """Calculate fair value for KELP using enhanced methods"""
        prices = self.trader_data[Product.KELP]["prices"]
        
        if len(prices) < 5:
            return self.params[Product.KELP]["avg_mid"]
            
        try:
            simple_ma = statistics.mean(prices[-10:])
            
            # Use numpy's weighted average with proper weights
            weights = list(range(1, min(21, len(prices) + 1)))
            price_window = prices[-len(weights):]
            
            weighted_ma = np.average(price_window, weights=weights)
            
            # More weight to recent prices
            return (weighted_ma * 0.7) + (simple_ma * 0.3)
        except Exception:
            # Fallback to last fair value
            return self.trader_data[Product.KELP]["last_fair_value"]
    
    def assess_kelp_trend(self):
        """Assess market trend for KELP"""
        prices = self.trader_data[Product.KELP]["prices"]
        
        if len(prices) < 5:
            return 0
        
        try:    
            short_term = np.polyfit(
                range(min(5, len(prices))), 
                prices[-min(5, len(prices)):], 
                1
            )[0]
            
            if len(prices) >= 10:
                medium_term = np.polyfit(
                    range(min(10, len(prices))), 
                    prices[-min(10, len(prices)):], 
                    1
                )[0]
                return np.clip((short_term * 0.7 + medium_term * 0.3) * 10, -1, 1)
            return np.clip(short_term * 10, -1, 1)
        except Exception:
            return 0
    
    def update_kelp_metrics(self, mid_price, spread, order_depth):
        """Update market metrics for KELP"""
        kelp_data = self.trader_data[Product.KELP]
        
        kelp_data["prices"].append(mid_price)
        kelp_data["spreads"].append(spread)
        
        # Calculate volumes
        total_buy_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
        total_sell_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
        kelp_data["volumes"].append(total_buy_volume + total_sell_volume)
        
        # Calculate volatility
        kelp_data["volatility"] = self.calculate_volatility(
            kelp_data["prices"], 
            self.params[Product.KELP]["volatility_window"]
        )
        
        # Assess trend
        kelp_data["trend"] = self.assess_kelp_trend()
        
        # Trim history
        if len(kelp_data["prices"]) > 100:
            kelp_data["prices"] = kelp_data["prices"][-50:]
            kelp_data["spreads"] = kelp_data["spreads"][-50:]
            kelp_data["volumes"] = kelp_data["volumes"][-50:]
    
    def adjust_kelp_parameters(self):
        """Adjust KELP trading parameters based on market conditions"""
        kelp_data = self.trader_data[Product.KELP]
        kelp_params = self.params[Product.KELP]
        
        volatility = kelp_data["volatility"]
        trend_strength = kelp_data["trend"]
        
        # Adjust aggression based on trend strength
        kelp_params["aggression_level"] = 0.3 + abs(trend_strength) * 0.5
        
        # Adjust spread multiplier based on volatility
        kelp_params["dynamic_spread_multiplier"] = 0.005 + (volatility / 2000.0)
        
        # Adjust profit threshold based on volatility and trend
        if abs(trend_strength) > 0.5:
            kelp_params["min_profit_threshold"] = max(0.1, 1.5 - abs(trend_strength)/2)
        else:
            kelp_params["min_profit_threshold"] = 1.5
    
    def enhanced_kelp_strategy(self, order_depth: OrderDepth, timestamp: int) -> List[Order]:
        """Enhanced strategy for KELP combining both implementations"""
        orders = []
        kelp_data = self.trader_data[Product.KELP]
        kelp_params = self.params[Product.KELP]
        
        current_position = kelp_data["position"]
        
        # Get market data
        buy_prices = list(order_depth.buy_orders.keys())
        sell_prices = list(order_depth.sell_orders.keys())
        
        if not buy_prices or not sell_prices:
            return orders
            
        best_bid = max(buy_prices)
        best_ask = min(sell_prices)
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Update market metrics
        self.update_kelp_metrics(mid_price, spread, order_depth)
        
        # Calculate fair value using enhanced method
        fair_value = self.calculate_kelp_fair_value()
        kelp_data["last_fair_value"] = fair_value
        kelp_params["fair_value"] = fair_value
        
        # Adjust strategy parameters
        self.adjust_kelp_parameters()
        
        # Check for emergency exit
        if self.need_kelp_emergency_exit(mid_price, current_position):
            if current_position > 0:
                emergency_size = current_position
                orders.append(Order(Product.KELP, best_bid, -emergency_size))
            elif current_position < 0:
                emergency_size = -current_position
                orders.append(Order(Product.KELP, best_ask, emergency_size))
            return orders
        
        # Combine strategies from both implementations
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Strategy 1: Take orders (from first implementation)
        take_orders, buy_order_volume, sell_order_volume = self.take_orders(
            Product.KELP,
            order_depth,
            fair_value,
            kelp_params["take_width"],
            current_position
        )
        orders.extend(take_orders)
        
        # Strategy 2: Clear orders (from first implementation)
        clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            Product.KELP,
            order_depth,
            fair_value,
            kelp_params["clear_width"],
            current_position,
            buy_order_volume,
            sell_order_volume
        )
        orders.extend(clear_orders)
        
        # Strategy 3: Aggressive orders (from second implementation)
        trend_strength = kelp_data["trend"]
        
        # More aggressive in downtrends (selling)
        if trend_strength < -0.3:
            sell_threshold = fair_value + kelp_params["min_profit_threshold"] * 0.7
            if best_bid > sell_threshold:
                max_sell_size = min(
                    int(kelp_params["max_order_size"] * kelp_params["aggression_level"]),
                    kelp_params["position_limit"] + current_position - sell_order_volume
                )
                bid_volume = order_depth.buy_orders.get(best_bid, 0)
                size = min(max_sell_size, int(bid_volume * kelp_params["liquidity_absorption_factor"]))
                if size > 0:
                    orders.append(Order(Product.KELP, best_bid, -size))
                    sell_order_volume += size
        
        # More aggressive in uptrends (buying)
        elif trend_strength > 0.3:
            buy_threshold = fair_value - kelp_params["min_profit_threshold"] * 0.7
            if best_ask < buy_threshold:
                max_buy_size = min(
                    int(kelp_params["max_order_size"] * kelp_params["aggression_level"]),
                    kelp_params["position_limit"] - current_position - buy_order_volume
                )
                ask_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                size = min(max_buy_size, int(ask_volume * kelp_params["liquidity_absorption_factor"]))
                if size > 0:
                    orders.append(Order(Product.KELP, best_ask, size))
                    buy_order_volume += size
        
        # Strategy 4: Market making (from first implementation)
        make_orders, _, _ = self.make_orders(
            Product.KELP,
            order_depth,
            fair_value,
            current_position,
            buy_order_volume,
            sell_order_volume,
            kelp_params["disregard_edge"],
            kelp_params["join_edge"],
            kelp_params["default_edge"],
            True,
            kelp_params["soft_position_limit"]
        )
        orders.extend(make_orders)
        
        return orders
    
    def need_kelp_emergency_exit(self, current_price, position):
        """Check if emergency exit is needed for KELP"""
        if position == 0:
            return False
            
        kelp_data = self.trader_data[Product.KELP]
        kelp_params = self.params[Product.KELP]
        
        if len(kelp_data["prices"]) < 10:
            return False
            
        try:
            # Calculate recent price movement
            highest_recent = max(kelp_data["prices"][-10:])
            price_change = (current_price - highest_recent) / highest_recent
            
            # Emergency exit if price drops significantly when long or rises when short
            if (position > 0 and price_change < kelp_params["panic_sell_threshold"]) or \
               (position < 0 and price_change > -kelp_params["panic_sell_threshold"]):
                return True
                
            return False
        except:
            return False
    
    ###############################
    # RESIN STRATEGY IMPLEMENTATION
    ###############################
    
    def resin_strategy(self, order_depth: OrderDepth, timestamp: int) -> List[Order]:
        """Strategy for RAINFOREST_RESIN using first implementation"""
        resin_data = self.trader_data[Product.RESIN]
        resin_params = self.params[Product.RESIN]
        
        current_position = resin_data["position"]
        fair_value = resin_params["fair_value"]
        
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Strategy 1: Take orders
        take_orders, buy_order_volume, sell_order_volume = self.take_orders(
            Product.RESIN,
            order_depth,
            fair_value,
            resin_params["take_width"],
            current_position
        )
        orders.extend(take_orders)
        
        # Strategy 2: Clear orders
        clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            Product.RESIN,
            order_depth,
            fair_value,
            resin_params["clear_width"],
            current_position,
            buy_order_volume,
            sell_order_volume
        )
        orders.extend(clear_orders)
        
        # Strategy 3: Market making
        make_orders, _, _ = self.make_orders(
            Product.RESIN,
            order_depth,
            fair_value,
            current_position,
            buy_order_volume,
            sell_order_volume,
            resin_params["disregard_edge"],
            resin_params["join_edge"],
            resin_params["default_edge"],
            True,
            resin_params["soft_position_limit"]
        )
        orders.extend(make_orders)
        
        return orders
    
    #################################
    # SQUID INK STRATEGY IMPLEMENTATION
    #################################
    
    def detect_squid_price_jumps(self, prices, threshold=20.0):
        """Detect sudden price movements for SQUID_INK"""
        if len(prices) < 2:
            return False
        
        squid_data = self.trader_data[Product.SQUID]
        price_change = abs(prices[-1] - prices[-2])
        if price_change > threshold:
            squid_data["price_jumps"].append({
                "timestamp": squid_data["last_timestamp"],
                "change": price_change,
                "direction": 1 if prices[-1] > prices[-2] else -1
            })
            return True
        return False
    
    def calculate_squid_fair_value(self):
        """Calculate fair value for SQUID_INK with volatility awareness"""
        squid_data = self.trader_data[Product.SQUID]
        squid_params = self.params[Product.SQUID]
        prices = squid_data["prices"]
        
        if len(prices) < 5:
            return squid_params["default_fair_value"]
        
        try:
            # Simple moving average
            simple_ma = statistics.mean(prices[-10:])
            
            # Weighted moving average
            weights = list(range(1, min(21, len(prices) + 1)))
            price_window = prices[-len(weights):]
            weighted_ma = np.average(price_window, weights=weights)
            
            # Mean reversion component
            volatility = squid_data["volatility"]
            current_price = prices[-1]
            
            # Calculate distance from moving average in terms of volatility
            z_score = (current_price - weighted_ma) / (volatility if volatility > 0 else 1)
            
            # Apply mean reversion when price is far from mean
            mean_reversion_adjustment = 0
            if abs(z_score) > squid_params["volatility_bands_width"]:
                reversion_direction = -1 if z_score > 0 else 1
                mean_reversion_adjustment = reversion_direction * volatility * squid_params["mean_reversion_strength"]
            
            # Combine indicators with mean reversion
            fair_value = weighted_ma * 0.7 + simple_ma * 0.3 + mean_reversion_adjustment
            
            # Stay within reasonable bounds based on price history
            min_price = squid_data["price_extremes"]["min"]
            max_price = squid_data["price_extremes"]["max"]
            return max(min_price, min(max_price, fair_value))
        except Exception:
            # Fallback to last fair value
            return squid_data["last_fair_value"]
            
    def adjust_squid_parameters(self):
                """Adjust SQUID_INK trading parameters based on market conditions"""
                squid_data = self.trader_data[Product.SQUID]
                squid_params = self.params[Product.SQUID]
                
                volatility = squid_data["volatility"]
                trend_strength = squid_data["trend"]
                
                # Adjust spread based on volatility
                squid_params["base_spread"] = 5.0 + (volatility / 100.0)
                
                # Adjust aggression based on trend strength and recent price jumps
                recent_jumps = len([j for j in squid_data["price_jumps"] if j["timestamp"] > squid_data["last_timestamp"] - 5])
                if recent_jumps > 0:
                    # Be more cautious after price jumps
                    squid_params["aggression_level"] = 0.1
                else:
                    # Normal aggression adjusted by trend
                    squid_params["aggression_level"] = 0.3 + abs(trend_strength) * 0.3
                
                # Adjust mean reversion strength based on volatility
                if volatility > 50:
                    squid_params["mean_reversion_strength"] = 0.8
                else:
                    squid_params["mean_reversion_strength"] = 0.6
                
                # Adjust order size based on volatility and position
                position_ratio = abs(squid_data["position"]) / squid_params["position_limit"]
                squid_params["max_order_size"] = max(1, int(5 * (1 - position_ratio) * (1 - min(1, volatility/100))))
            
    def advanced_squid_strategy(self, order_depth: OrderDepth, timestamp: int) -> List[Order]:
            """Advanced strategy for SQUID_INK with volatility awareness and advanced order placement"""
            orders = []
            squid_data = self.trader_data[Product.SQUID]
            squid_params = self.params[Product.SQUID]
            
            current_position = squid_data["position"]
            squid_data["last_timestamp"] = timestamp
            
            # Get market data
            buy_prices = list(order_depth.buy_orders.keys())
            sell_prices = list(order_depth.sell_orders.keys())
            
            if not buy_prices or not sell_prices:
                return orders
                
            best_bid = max(buy_prices)
            best_ask = min(sell_prices)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            # Update market metrics
            self.update_squid_metrics(mid_price, spread, order_depth)
            
            # Calculate fair value
            fair_value = self.calculate_squid_fair_value()
            squid_data["last_fair_value"] = fair_value
            
            # Adjust strategy parameters
            self.adjust_squid_parameters()
            
            # Check for recent price jumps to determine if we should be more defensive
            recent_jumps = len([j for j in squid_data["price_jumps"] if j["timestamp"] > timestamp - 5])
            is_volatile_market = recent_jumps > 0 or squid_data["volatility"] > 50
            
            # Strategy components
            buy_order_volume = 0
            sell_order_volume = 0
            
            # 1. Opportunistic taking of mispriced orders (aggressive)
            if not is_volatile_market:
                # Buy underpriced offers
                for price in sorted(sell_prices):
                    if price < fair_value - squid_params["min_profit_threshold"]:
                        available_to_buy = min(
                            squid_params["position_limit"] - current_position - buy_order_volume,
                            abs(order_depth.sell_orders[price]),
                            squid_params["max_order_size"]
                        )
                        if available_to_buy > 0:
                            orders.append(Order(Product.SQUID, price, available_to_buy))
                            buy_order_volume += available_to_buy
                    else:
                        break
                
                # Sell overpriced bids
                for price in sorted(buy_prices, reverse=True):
                    if price > fair_value + squid_params["min_profit_threshold"]:
                        available_to_sell = min(
                            squid_params["position_limit"] + current_position - sell_order_volume,
                            order_depth.buy_orders[price],
                            squid_params["max_order_size"]
                        )
                        if available_to_sell > 0:
                            orders.append(Order(Product.SQUID, price, -available_to_sell))
                            sell_order_volume += available_to_sell
                    else:
                        break
            
            # 2. Position management (defensive)
            if abs(current_position) > squid_params["soft_position_limit"]:
                if current_position > 0:
                    # Try to reduce long position
                    target_price = max(fair_value, best_bid)
                    available_to_sell = min(
                        current_position,
                        squid_params["max_order_size"] * 2
                    )
                    if available_to_sell > 0:
                        orders.append(Order(Product.SQUID, target_price, -available_to_sell))
                        sell_order_volume += available_to_sell
                else:
                    # Try to reduce short position
                    target_price = min(fair_value, best_ask)
                    available_to_buy = min(
                        -current_position,
                        squid_params["max_order_size"] * 2
                    )
                    if available_to_buy > 0:
                        orders.append(Order(Product.SQUID, target_price, available_to_buy))
                        buy_order_volume += available_to_buy
            
            # 3. Market making around fair value
            if not is_volatile_market:
                position_ratio = abs(current_position) / squid_params["position_limit"]
                dynamic_spread = squid_params["base_spread"] * (1 + position_ratio + squid_data["volatility"]/200)
                
                bid_price = round(fair_value - dynamic_spread)
                ask_price = round(fair_value + dynamic_spread)
                
                # Skip market making if our fair value is too far from market
                market_deviation = abs(mid_price - fair_value) / fair_value
                if market_deviation < 0.02:  # Only make markets if we're within 2% of market
                    # Place buy order (with position-aware sizing)
                    buy_size = max(1, int(squid_params["max_order_size"] * (1 - position_ratio/2)))
                    available_to_buy = squid_params["position_limit"] - current_position - buy_order_volume
                    if available_to_buy >= buy_size:
                        orders.append(Order(Product.SQUID, bid_price, buy_size))
                        buy_order_volume += buy_size
                    
                    # Place sell order (with position-aware sizing)
                    sell_size = max(1, int(squid_params["max_order_size"] * (1 - position_ratio/2)))
                    available_to_sell = squid_params["position_limit"] + current_position - sell_order_volume
                    if available_to_sell >= sell_size:
                        orders.append(Order(Product.SQUID, ask_price, -sell_size))
                        sell_order_volume += sell_size
            
            # Track P&L (approximation)
            for order in orders:
                if order.quantity > 0:  # Buying
                    squid_data["profit_loss"] -= order.price * order.quantity
                else:  # Selling
                    squid_data["profit_loss"] += order.price * abs(order.quantity)
            
            squid_data["trade_count"] += len(orders)
            return orders
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Main trading logic that runs for each product.
        """
        result = {}
        traderData = jsonpickle.encode(self.trader_data)
        
        # Process each available product
        for product in state.order_depths.keys():
            if product == Product.KELP:
                # Update position from previous iteration
                self.trader_data[Product.KELP]["position"] = state.position.get(product, 0)
                
                # Run KELP strategy
                result[product] = self.enhanced_kelp_strategy(
                    state.order_depths[product],
                    state.timestamp
                )
                
            elif product == Product.RESIN:
                # Update position from previous iteration
                self.trader_data[Product.RESIN]["position"] = state.position.get(product, 0)
                
                # Run RESIN strategy
                result[product] = self.resin_strategy(
                    state.order_depths[product],
                    state.timestamp
                )
                
            elif product == Product.SQUID:
                # Update position from previous iteration
                self.trader_data[Product.SQUID]["position"] = state.position.get(product, 0)
                
                # Run SQUID strategy
                result[product] = self.advanced_squid_strategy(
                    state.order_depths[product],
                    state.timestamp
                )
        
        # Log the current state if needed
        logger.flush(state, result, 0, traderData)
        
        return result