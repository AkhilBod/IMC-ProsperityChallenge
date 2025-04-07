from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle
import math

class Trader:
    def __init__(self):
        # Strategy parameters
        self.ema_periods = 200
        self.kelp_std_dev_multiplier = 2.5
        self.resin_std_dev_multiplier = 1.0
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], List, str]:
        # Print trader data and observations as required - MATCHING GOLD STANDARD FORMAT
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        # Initialize result containers
        result = {}
        conversions = []
        
        # Load previous state if available
        trader_state = self.initialize_state(state.traderData)
        
        # Process each product in order depths - MATCHING GOLD STANDARD PATTERN
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            # Skip if no orders to process
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue
            
            # Process the product based on type
            if product == "KELP":
                self.process_kelp(state, trader_state, product, order_depth, orders)
            elif product == "RAINFOREST_RESIN":
                self.process_rainforest_resin(state, trader_state, product, order_depth, orders)
            else:
                # Default handling for any other product
                self.process_default(state, trader_state, product, order_depth, orders)
            
            # Store orders for this product
            result[product] = orders
        
        # Save updated state
        trader_data = jsonpickle.encode(trader_state)
        
        return result, conversions, trader_data
    
    def process_kelp(self, state, trader_state, product, order_depth, orders):
        """Process KELP trading strategy"""
        
        # Initialize product data if not already present
        if product not in trader_state["historical_prices"]:
            trader_state["historical_prices"][product] = []
            trader_state["ema_values"][product] = None
        
        # Calculate mid price from order book
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]
        mid_price = (best_bid + best_ask) / 2
        
        # Store historical price
        trader_state["historical_prices"][product].append(mid_price)
        
        # Limit history length
        if len(trader_state["historical_prices"][product]) > self.ema_periods:
            trader_state["historical_prices"][product] = trader_state["historical_prices"][product][-self.ema_periods:]
        
        # Skip if not enough data yet
        if len(trader_state["historical_prices"][product]) < self.ema_periods:
            return
        
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
        
        # Calculate Keltner Channel bands with 2.5 std dev for KELP
        upper_band = trader_state["ema_values"][product] + (std_dev * self.kelp_std_dev_multiplier)
        lower_band = trader_state["ema_values"][product] - (std_dev * self.kelp_std_dev_multiplier)
        

        # Get current position for this product
        current_position = state.position.get(product, 0)
        
        # Get position limit for this product
        position_limit = 50  # Adjust based on actual limits
        
        # Buy signal - price touches lower band
        if best_ask <= lower_band:
            # Calculate available position size
            available_position = position_limit - current_position
            if available_position > 0:
                # Trade with max lot size
                buy_quantity = min(available_position, abs(best_ask_amount))
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
                    # Log buy order - MATCHES GOLD STANDARD FORMAT
                    print("BUY", str(buy_quantity) + "x", best_ask)
                
        # Sell signal - price touches upper band
        elif best_bid >= upper_band:
            # Calculate available position size
            available_position = current_position + position_limit
            if available_position > 0:
                # Trade with max lot size
                sell_quantity = min(available_position, best_bid_amount)
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
                    # Log sell order - MATCHES GOLD STANDARD FORMAT
                    print("SELL", str(sell_quantity) + "x", best_bid)
    
    def process_rainforest_resin(self, state, trader_state, product, order_depth, orders):
        """Process RAINFOREST_RESIN trading strategy"""
        
        # Initialize product data if not already present
        if product not in trader_state["historical_prices"]:
            trader_state["historical_prices"][product] = []
            trader_state["ema_values"][product] = None
        
        # Calculate mid price from order book
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]
        mid_price = (best_bid + best_ask) / 2
        
        # Store historical price
        trader_state["historical_prices"][product].append(mid_price)
        
        # Limit history length
        if len(trader_state["historical_prices"][product]) > self.ema_periods:
            trader_state["historical_prices"][product] = trader_state["historical_prices"][product][-self.ema_periods:]
        
        # Skip if not enough data yet
        if len(trader_state["historical_prices"][product]) < self.ema_periods:
            return
        
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
        
        # Calculate Keltner Channel bands with 1.0 std dev for RAINFOREST_RESIN
        upper_band = trader_state["ema_values"][product] + (std_dev * self.resin_std_dev_multiplier)
        lower_band = trader_state["ema_values"][product] - (std_dev * self.resin_std_dev_multiplier)
        
        # Get current position for this product
        current_position = state.position.get(product, 0)
        
        # Get position limit for this product
        position_limit = 50  # Adjust based on actual limits
        
        # Buy signal - price touches lower band
        if best_ask <= lower_band:
            # Calculate available position size
            available_position = position_limit - current_position
            if available_position > 0:
                # Trade with max lot size
                buy_quantity = min(available_position, abs(best_ask_amount))
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
                    # Log buy order - MATCHES GOLD STANDARD FORMAT
                    print("BUY", str(buy_quantity) + "x", best_ask)
                
        # Sell signal - price touches upper band
        elif best_bid >= upper_band:
            # Calculate available position size
            available_position = current_position + position_limit
            if available_position > 0:
                # Trade with max lot size
                sell_quantity = min(available_position, best_bid_amount)
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
                    # Log sell order - MATCHES GOLD STANDARD FORMAT
                    print("SELL", str(sell_quantity) + "x", best_bid)
    
    def process_default(self, state, trader_state, product, order_depth, orders):
        """Default processing for any other product"""
        
        # Calculate mid price from order book
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
    
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