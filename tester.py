from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import statistics
import math
import json

class Trader:
    def __init__(self):
        # Initialize state variables
        self.kelp_position = 0
        self.position_limit = 20  # Adjust based on actual position limit
        self.kelp_trades = []
        self.kelp_prices = []
        self.kelp_fair_values = []
        self.ema_short = None
        self.ema_long = None
        self.short_window = 20
        self.long_window = 50
        self.volatility = 10 # From data analysis
        self.mean_price = 2000.84  # From data analysis
        self.spread_mean = 1.98  # From data analysis
        self.day = 0  # Track which day we're on

    def update_ema(self, price, ema, window):
        """Update exponential moving average"""
        if ema is None:
            return price
        else:
            alpha = 2 / (window + 1)
            return price * alpha + ema * (1 - alpha)
    
    def calculate_fair_value(self, order_depth, market_trades, own_trades, timestamp):
        """Calculate fair value for KELP based on market data"""
        # Extract current bid-ask 
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        
        # Calculate mid price if both sides are available
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
        elif best_bid:
            mid_price = best_bid + self.spread_mean / 2
        elif best_ask:
            mid_price = best_ask - self.spread_mean / 2
        else:
            # No orders, use recent fair values or default
            if len(self.kelp_fair_values) > 0:
                mid_price = self.kelp_fair_values[-1]
            else:
                mid_price = self.mean_price

        # Add recent trades to our price history
        if market_trades:
            for trade in market_trades:
                self.kelp_prices.append(trade.price)

        if own_trades:
            for trade in own_trades:
                self.kelp_prices.append(trade.price)
        
        # Limit length of price history
        if len(self.kelp_prices) > self.long_window:
            self.kelp_prices = self.kelp_prices[-self.long_window:]

        # Update EMAs if we have prices
        if self.kelp_prices:
            self.ema_short = self.update_ema(mid_price, self.ema_short, self.short_window)
            self.ema_long = self.update_ema(mid_price, self.ema_long, self.long_window)
        
        # Adjust fair value based on order book imbalance
        total_bid_volume = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
        total_ask_volume = abs(sum(order_depth.sell_orders.values())) if order_depth.sell_orders else 0
        
        # Calculate order book imbalance
        if total_bid_volume + total_ask_volume > 0:
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            # Adjust fair value based on imbalance (more buyers than sellers = higher price)
            fair_value = mid_price + (imbalance * self.volatility * 0.2)
        else:
            fair_value = mid_price
        
        # If we have enough data for trend detection
        if self.ema_short is not None and self.ema_long is not None:
            # Adjust for trend (small effect)
            trend_factor = (self.ema_short - self.ema_long) / self.mean_price * 100
            fair_value += trend_factor
        
        # Store fair value for next iteration
        self.kelp_fair_values.append(fair_value)
        if len(self.kelp_fair_values) > 100:
            self.kelp_fair_values = self.kelp_fair_values[-100:]
            
        # For KELP, our data shows gradually increasing prices across days
        # Day 0: ~2011, Day 1: ~2023, Day 2: ~2034 (average increase of ~11-12 per day)
        # Add a slight upward bias based on our analysis of price trends
        daily_trend_adjustment = self.day * 11.5
        fair_value += daily_trend_adjustment
            
        return fair_value

    def calculate_optimal_orders(self, fair_value, order_depth, position):
        """Calculate optimal orders based on fair value and market data"""
        orders = []
        
        # Update current position
        self.kelp_position = position
        remaining_long = self.position_limit - self.kelp_position
        remaining_short = self.position_limit + self.kelp_position
        
        # Calculate uncertainty range around fair value
        uncertainty = self.volatility * 0.2  # Tighter range for more aggressive trading
        upper_bound = fair_value + uncertainty
        lower_bound = fair_value - uncertainty
        
        # Process sell side (we buy)
        potential_buys = []
        for price, volume in sorted(order_depth.sell_orders.items()):
            vol = abs(volume)  # Volume is negative in sell orders
            if price < fair_value:  # Good price to buy (below fair value)
                # Calculate how much profit we expect to make
                expected_profit = fair_value - price
                # Higher profit = higher priority
                potential_buys.append((price, vol, expected_profit))
        
        # Sort by expected profit (highest first)
        potential_buys.sort(key=lambda x: x[2], reverse=True)
        
        # Process buy side (we sell)
        potential_sells = []
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            vol = volume  # Volume is positive in buy orders
            if price > fair_value:  # Good price to sell (above fair value)
                # Calculate how much profit we expect to make
                expected_profit = price - fair_value
                # Higher profit = higher priority
                potential_sells.append((price, vol, expected_profit))
        
        # Sort by expected profit (highest first)
        potential_sells.sort(key=lambda x: x[2], reverse=True)
        
        # Execute buys (highest profit first, respecting position limits)
        buy_qty = 0
        for price, volume, profit in potential_buys:
            if buy_qty >= remaining_long:
                break
                
            # Buy as much as we can without exceeding position limit
            qty_to_buy = min(volume, remaining_long - buy_qty)
            if qty_to_buy > 0:
                orders.append(Order("KELP", price, qty_to_buy))
                buy_qty += qty_to_buy
        
        # Execute sells (highest profit first, respecting position limits)
        sell_qty = 0
        for price, volume, profit in potential_sells:
            if sell_qty >= remaining_short:
                break
                
            # Sell as much as we can without exceeding position limit
            qty_to_sell = min(volume, remaining_short - sell_qty)
            if qty_to_sell > 0:
                orders.append(Order("KELP", price, -qty_to_sell))
                sell_qty += qty_to_sell
                
        # If we didn't find any immediate profitable trades, place limit orders
        if not orders and remaining_long > 0:
            # Place a buy order just below the market
            bid_price = max(order_depth.buy_orders.keys()) + 1 if order_depth.buy_orders else (fair_value - 1)
            # Ensure we're not buying above our upper bound
            if bid_price < lower_bound:
                orders.append(Order("KELP", bid_price, min(5, remaining_long)))
                
        if not orders and remaining_short > 0:
            # Place a sell order just above the market
            ask_price = min(order_depth.sell_orders.keys()) - 1 if order_depth.sell_orders else (fair_value + 1)
            # Ensure we're not selling below our lower bound
            if ask_price > upper_bound:
                orders.append(Order("KELP", ask_price, -min(5, remaining_short)))
                
        return orders

    def run(self, state: TradingState):
        """
        Main method called by the platform to get our orders.
        """
        # Initialize the results dict we'll return
        result = {}
        
        # Deserialize trader data if available
        if state.traderData and state.traderData != "":
            try:
                data = json.loads(state.traderData)
                self.kelp_position = data.get("kelp_position", 0)
                self.kelp_fair_values = data.get("kelp_fair_values", [])
                self.ema_short = data.get("ema_short", None)
                self.ema_long = data.get("ema_long", None)
                self.day = data.get("day", 0)
                # Try to detect which day we're on based on average price
                if state.timestamp == 0:  # Beginning of a new day
                    if state.market_trades and "KELP" in state.market_trades and state.market_trades["KELP"]:
                        avg_price = sum(t.price for t in state.market_trades["KELP"]) / len(state.market_trades["KELP"])
                        if avg_price > 2030:
                            self.day = 0  # Most recent day
                        elif avg_price > 2015:
                            self.day = -1  # Middle day
                        else:
                            self.day = -2  # Earliest day
            except Exception as e:
                print(f"Error deserializing trader data: {e}")
        
        # Process each product
        for product in state.order_depths:
            if product == "KELP":
                # Get the order depth for this product
                order_depth = state.order_depths[product]
                
                # Get position
                position = state.position.get(product, 0)
                
                # Get recent trades
                market_trades = state.market_trades.get(product, [])
                own_trades = state.own_trades.get(product, [])
                
                # Calculate fair value
                fair_value = self.calculate_fair_value(order_depth, market_trades, own_trades, state.timestamp)
                
                # Calculate and place orders
                orders = self.calculate_optimal_orders(fair_value, order_depth, position)
                
                # Add orders to result
                result[product] = orders
                
                # Debug logs
                print(f"KELP position: {position}, fair value: {fair_value}")
                print(f"Orders: {orders}")
        
        # Serialize trader data
        trader_data = json.dumps({
            "kelp_position": self.kelp_position,
            "kelp_fair_values": self.kelp_fair_values[-100:] if self.kelp_fair_values else [],
            "ema_short": self.ema_short,
            "ema_long": self.ema_long,
            "day": self.day,
        })
        
        # No conversions needed for KELP trading
        conversions = 0
        
        return result, conversions, trader_data
