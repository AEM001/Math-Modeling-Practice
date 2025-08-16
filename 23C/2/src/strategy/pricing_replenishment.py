"""
Heuristic Pricing and Replenishment Strategy Module (Stage 6)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

class PricingReplenishmentStrategy:
    def __init__(self):
        self.logger = setup_logger('pricing_replenishment')
        self.pricing_results = []
        
    def calculate_base_markup_pricing(self, cost: float, base_markup: float = None) -> float:
        """Calculate base cost-plus pricing"""
        if base_markup is None:
            base_markup = Config.DEFAULT_MARKUP
        
        price = cost * (1 + base_markup)
        return price
    
    def apply_pricing_constraints(self, price: float, cost: float, 
                                previous_price: float = None) -> float:
        """Apply business constraints to pricing"""
        
        # Price-to-cost ratio constraints
        min_price = cost * Config.MIN_PRICE_COST_RATIO
        max_price = cost * Config.MAX_PRICE_COST_RATIO
        
        price = np.clip(price, min_price, max_price)
        
        # Daily price change constraint
        if previous_price is not None and previous_price > 0:
            max_change = previous_price * Config.MAX_DAILY_PRICE_CHANGE
            min_allowed = previous_price - max_change
            max_allowed = previous_price + max_change
            price = np.clip(price, min_allowed, max_allowed)
        
        # Round to reasonable precision (0.1 yuan)
        price = round(price, 1)
        
        return price
    
    def optimize_markup_in_range(self, cost: float, predicted_demand: float, 
                                price_elasticity: float, previous_price: float = None) -> Tuple[float, float]:
        """Optimize markup within allowed range for revenue maximization"""
        
        best_price = None
        best_revenue = -np.inf
        best_markup = Config.DEFAULT_MARKUP
        
        # Grid search over markup range
        markup_range = np.arange(Config.MIN_MARKUP, Config.MAX_MARKUP + 0.01, 0.02)
        
        for markup in markup_range:
            price = self.calculate_base_markup_pricing(cost, markup)
            price = self.apply_pricing_constraints(price, cost, previous_price)
            
            # Estimate demand response using price elasticity
            if not np.isnan(price_elasticity) and price_elasticity != 0:
                # Use elasticity to adjust demand
                base_price = cost * (1 + Config.DEFAULT_MARKUP)
                price_change_pct = (price - base_price) / base_price
                demand_change_pct = price_elasticity * price_change_pct
                adjusted_demand = predicted_demand * (1 + demand_change_pct)
                adjusted_demand = max(0, adjusted_demand)  # Non-negative demand
            else:
                # Use simple linear approximation if elasticity not available
                base_price = cost * (1 + Config.DEFAULT_MARKUP)
                demand_adjustment = max(0, 1 - 0.5 * (price - base_price) / base_price)
                adjusted_demand = predicted_demand * demand_adjustment
            
            # Calculate expected revenue
            revenue = price * adjusted_demand
            
            if revenue > best_revenue:
                best_revenue = revenue
                best_price = price
                best_markup = markup
        
        return best_price if best_price is not None else self.calculate_base_markup_pricing(cost), best_markup
    
    def calculate_replenishment_quantity(self, predicted_demand: float, 
                                       prediction_std: float, 
                                       loss_rate: float = 0.1,
                                       service_level: float = None) -> float:
        """Calculate replenishment quantity using service level approach"""
        
        if service_level is None:
            service_level = Config.SERVICE_LEVEL
        
        # Get z-score for service level
        if service_level >= 0.975:
            z_score = Config.Z_SCORE_975
        elif service_level >= 0.95:
            z_score = Config.Z_SCORE_95
        else:
            z_score = Config.Z_SCORE_90
        
        # Safety stock calculation
        safety_stock = z_score * prediction_std
        
        # Total replenishment including safety stock and loss rate
        replenish_qty = (predicted_demand + safety_stock) / (1 - loss_rate)
        
        # Round up to ensure sufficient stock
        replenish_qty = np.ceil(replenish_qty)
        
        return max(0, replenish_qty)
    
    def estimate_category_loss_rate(self, historical_df: pd.DataFrame, category: str) -> float:
        """Estimate loss rate for category from historical data"""
        
        category_data = historical_df[historical_df['分类名称'] == category].copy()
        
        if len(category_data) == 0:
            return 0.1  # Default 10% loss rate
        
        # Use median loss rate from historical data
        if '损耗率(%)' in category_data.columns:
            loss_rates = category_data['损耗率(%)'] / 100
            median_loss_rate = loss_rates.median()
            return np.clip(median_loss_rate, 0.05, 0.3)  # Bound between 5% and 30%
        else:
            return 0.1  # Default
    
    def generate_pricing_strategy(self, forecast_df: pd.DataFrame, 
                                historical_df: pd.DataFrame,
                                price_elasticities: Dict[str, float]) -> pd.DataFrame:
        """Generate pricing strategy for forecast period"""
        
        self.logger.info("Generating pricing strategy...")
        
        pricing_results = []
        previous_prices = {}  # Track previous day prices for each category
        
        # Sort by date to ensure proper sequence
        forecast_df = forecast_df.sort_values('销售日期')
        
        for _, row in forecast_df.iterrows():
            category = row['分类名称']
            forecast_date = row['销售日期']
            predicted_demand = row['predicted_demand']
            estimated_cost = row['estimated_cost']
            prediction_std = row.get('prediction_std', predicted_demand * 0.2)  # Default 20% CV
            
            # Get price elasticity for this category
            elasticity = price_elasticities.get(category, np.nan)
            
            # Get previous price for constraint
            previous_price = previous_prices.get(category, None)
            
            # Optimize pricing
            optimal_price, optimal_markup = self.optimize_markup_in_range(
                estimated_cost, predicted_demand, elasticity, previous_price
            )
            
            # Calculate replenishment quantity
            loss_rate = self.estimate_category_loss_rate(historical_df, category)
            replenish_qty = self.calculate_replenishment_quantity(
                predicted_demand, prediction_std, loss_rate
            )
            
            # Store result
            pricing_result = {
                'date': forecast_date,
                'category': category,
                'price': optimal_price,
                'replenish_qty': replenish_qty,
                'demand_pred': predicted_demand,
                'cost_est': estimated_cost,
                'markup': optimal_markup,
                'service_level': Config.SERVICE_LEVEL,
                'loss_rate': loss_rate,
                'prediction_std': prediction_std,
                'price_elasticity': elasticity,
                'expected_revenue': optimal_price * predicted_demand,
                'expected_profit': (optimal_price - estimated_cost) * predicted_demand
            }
            
            pricing_results.append(pricing_result)
            
            # Update previous price for this category
            previous_prices[category] = optimal_price
        
        pricing_df = pd.DataFrame(pricing_results)
        
        self.logger.info(f"Generated pricing strategy for {len(pricing_df)} category-day combinations")
        return pricing_df
    
    def validate_strategy(self, pricing_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate pricing and replenishment strategy"""
        
        self.logger.info("Validating strategy...")
        
        validation_stats = {
            'total_combinations': len(pricing_df),
            'avg_markup': pricing_df['markup'].mean(),
            'markup_std': pricing_df['markup'].std(),
            'min_markup': pricing_df['markup'].min(),
            'max_markup': pricing_df['markup'].max(),
            'avg_service_level': pricing_df['service_level'].mean(),
            'total_expected_revenue': pricing_df['expected_revenue'].sum(),
            'total_expected_profit': pricing_df['expected_profit'].sum(),
            'avg_profit_margin': (pricing_df['expected_profit'] / pricing_df['expected_revenue']).mean()
        }
        
        # Check constraint violations
        violations = {
            'negative_prices': (pricing_df['price'] <= 0).sum(),
            'negative_quantities': (pricing_df['replenish_qty'] < 0).sum(),
            'extreme_markups': ((pricing_df['markup'] < Config.MIN_MARKUP) | 
                               (pricing_df['markup'] > Config.MAX_MARKUP)).sum()
        }
        
        validation_stats['violations'] = violations
        
        self.logger.info("=== Strategy Validation Summary ===")
        self.logger.info(f"Total category-day combinations: {validation_stats['total_combinations']}")
        self.logger.info(f"Average markup: {validation_stats['avg_markup']:.3f}")
        self.logger.info(f"Markup range: [{validation_stats['min_markup']:.3f}, {validation_stats['max_markup']:.3f}]")
        self.logger.info(f"Total expected revenue: ¥{validation_stats['total_expected_revenue']:,.2f}")
        self.logger.info(f"Total expected profit: ¥{validation_stats['total_expected_profit']:,.2f}")
        self.logger.info(f"Average profit margin: {validation_stats['avg_profit_margin']:.3f}")
        
        # Log violations
        total_violations = sum(violations.values())
        if total_violations > 0:
            self.logger.warning(f"Found {total_violations} constraint violations:")
            for violation_type, count in violations.items():
                if count > 0:
                    self.logger.warning(f"  {violation_type}: {count}")
        else:
            self.logger.info("No constraint violations detected")
        
        return validation_stats
    
    def save_strategy(self, pricing_df: pd.DataFrame):
        """Save pricing and replenishment strategy"""
        
        output_path = Config.get_output_path(Config.PRICING_REPLENISHMENT_FILE)
        pricing_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"Pricing and replenishment strategy saved to: {output_path}")
    
    def run(self, forecast_df: pd.DataFrame, historical_df: pd.DataFrame, 
           price_elasticities: Dict[str, float]) -> pd.DataFrame:
        """Run complete pricing and replenishment strategy"""
        
        self.logger.info("Starting pricing and replenishment strategy...")
        
        # Generate strategy
        pricing_df = self.generate_pricing_strategy(forecast_df, historical_df, price_elasticities)
        
        # Validate strategy
        validation_stats = self.validate_strategy(pricing_df)
        
        # Save strategy
        self.save_strategy(pricing_df)
        
        self.logger.info("Pricing and replenishment strategy completed successfully!")
        return pricing_df

if __name__ == "__main__":
    # This would be called from the main pipeline
    pass