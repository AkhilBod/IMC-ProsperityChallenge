from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import statistics
import jsonpickle

class Trader:
    def __init__(self):
        self.product = "KELP"
        self.window_size = 30
        self.price_history = []
        self.position_history = []
        self.z_score_history = []
        
        # Strategy parameters
        self.buy_threshold = -1.0  # Buy when z-score is below this threshold
        self.sell_threshold = 1.0  # Sell when z-score is above this threshold
        self.exit_threshold = 0.5  # Exit positions when z-score is between -exit and +exit
        self.position_limit = 70   # Maximum position size (adjust based on actual limits)
        self.order_size = 10       # Base order size
        
        # For tracking current position
        self.current_position = 0
        
    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from order depth."""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        
        # Fallback if we don't have both sides
        if len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
        else:
            # No orders available, use last price if available
            if len(self.price_history) > 0:
                return self.price_history[-1]
            return None  # Cannot determine price
            
    def calculate_z_score(self) -> float:
        """Calculate z-score based on current price and historical window."""
        if len(self.price_history) < self.window_size:
            return 0  # Not enough data
            
        window = self.price_history[-self.window_size:]
        mean = statistics.mean(window)
        stdev = statistics.stdev(window)
        
        if stdev == 0:
            return 0  # Avoid division by zero
            
        current_price = self.price_history[-1]
        z_score = (current_price - mean) / stdev
        return z_score
        
    def determine_order_size(self, z_score: float, current_position: int) -> int:
        """Determine order size based on z-score and current position."""
        # Base order size adjusted by z-score magnitude
        size_multiplier = min(2.0, abs(z_score))
        base_size = round(self.order_size * size_multiplier)
        
        # Adjust for remaining position capacity
        if z_score < self.buy_threshold:  # Buying
            return min(base_size, self.position_limit - current_position)
        elif z_score > self.sell_threshold:  # Selling
            return -min(base_size, current_position + self.position_limit)
        else:  # Exiting positions
            if current_position > 0:  # We're long, so sell
                return -min(base_size, current_position)
            elif current_position < 0:  # We're short, so buy
                return min(base_size, -current_position)
            return 0  # No position to exit
            
    def run(self, state: TradingState):
        """Main trading logic."""
        print(f"Timestamp: {state.timestamp}, Trader Data: {state.traderData}")
        
        # Initialize result dict and trader data
        result = {}
        traderData = ""
        
        # Restore state if available
        if state.traderData != "":
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.price_history = saved_state.get("price_history", self.price_history)
                self.position_history = saved_state.get("position_history", self.position_history)
                self.z_score_history = saved_state.get("z_score_history", self.z_score_history)
            except Exception as e:
                print(f"Error restoring state: {e}")
        
        # Update current position
        if self.product in state.position:
            self.current_position = state.position[self.product]
        else:
            self.current_position = 0
            
        self.position_history.append(self.current_position)
        print(f"Current position: {self.current_position}")
        
        # Check if we have order depth for our product
        if self.product not in state.order_depths:
            print(f"No order depth for {self.product}")
            traderData = jsonpickle.encode({
                "price_history": self.price_history,
                "position_history": self.position_history,
                "z_score_history": self.z_score_history
            })
            return {}, 0, traderData
            
        order_depth = state.order_depths[self.product]
        
        # Calculate mid price and update history
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is not None:
            self.price_history.append(mid_price)
            print(f"Mid price: {mid_price}")
        else:
            print("Could not calculate mid price")
            traderData = jsonpickle.encode({
                "price_history": self.price_history,
                "position_history": self.position_history,
                "z_score_history": self.z_score_history
            })
            return {}, 0, traderData
            
        # Calculate z-score
        z_score = self.calculate_z_score()
        self.z_score_history.append(z_score)
        print(f"Z-score: {z_score}")
        
        # Determine trading action based on z-score
        orders = []
        
        # Check for direct arbitrage opportunity first
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            best_ask_volume = -order_depth.sell_orders[best_ask]  # Convert to positive
            
            if best_bid > best_ask:  # Arbitrage opportunity
                print(f"Arbitrage opportunity: Buy@{best_ask} - Sell@{best_bid}")
                # Calculate trade volume
                arb_volume = min(best_bid_volume, best_ask_volume, 10)  # Limit to 10 units for safety
                # Buy at ask, sell at bid
                orders.append(Order(self.product, best_ask, arb_volume))
                orders.append(Order(self.product, best_bid, -arb_volume))
            else:
                # No arbitrage, trade based on mean reversion strategy
                if len(self.price_history) >= self.window_size:  # Only trade when we have enough history
                    # Trading logic based on z-score
                    if z_score < self.buy_threshold and self.current_position < self.position_limit:
                        # Price is below threshold, buy signal
                        buy_price = best_ask  # Buy at ask price
                        buy_quantity = self.determine_order_size(z_score, self.current_position)
                        if buy_quantity > 0:
                            orders.append(Order(self.product, buy_price, buy_quantity))
                            print(f"BUY signal: z-score={z_score}, price={buy_price}, qty={buy_quantity}")
                            
                    elif z_score > self.sell_threshold and self.current_position > -self.position_limit:
                        # Price is above threshold, sell signal
                        sell_price = best_bid  # Sell at bid price
                        sell_quantity = self.determine_order_size(z_score, self.current_position)
                        if sell_quantity < 0:  # Ensure quantity is negative for sell orders
                            orders.append(Order(self.product, sell_price, sell_quantity))
                            print(f"SELL signal: z-score={z_score}, price={sell_price}, qty={sell_quantity}")
                            
                    elif abs(z_score) < self.exit_threshold and self.current_position != 0:
                        # Price is near mean, exit positions
                        exit_quantity = self.determine_order_size(z_score, self.current_position)
                        if exit_quantity != 0:
                            exit_price = best_ask if exit_quantity > 0 else best_bid
                            orders.append(Order(self.product, exit_price, exit_quantity))
                            print(f"EXIT signal: z-score={z_score}, price={exit_price}, qty={exit_quantity}")
        
        result[self.product] = orders
        
        # Save the state
        traderData = jsonpickle.encode({
            "price_history": self.price_history,
            "position_history": self.position_history,
            "z_score_history": self.z_score_history
        })
        
        return result, 0, traderData