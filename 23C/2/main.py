#!/usr/bin/env python3
"""
Main Pipeline for Vegetable Pricing and Replenishment Decision
2023 Mathematical Contest in Modeling - Problem C, Question 2

This script implements the complete 7-stage pipeline for optimizing
vegetable category pricing and replenishment strategies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config
from src.utils.logger import setup_logger
from src.data_processing.data_cleaner import DataCleaner
from src.feature_engineering.feature_builder import FeatureBuilder
from src.modeling.demand_models import DemandModeler
from src.modeling.backtest import BacktestEvaluator
from src.forecasting.demand_forecast import DemandForecaster
from src.strategy.pricing_replenishment import PricingReplenishmentStrategy
from src.visualization.report_generator import ReportGenerator

class VegetablePricingPipeline:
    def __init__(self):
        self.logger = setup_logger('main_pipeline')
        self.results = {}
        
    def setup_output_directories(self):
        """Create output directories if they don't exist"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        self.logger.info("Output directories created/verified")
    
    def stage_1_data_cleaning(self):
        """Stage 1: Data Audit and Cleaning"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: DATA AUDIT AND CLEANING")
        self.logger.info("=" * 60)
        
        cleaner = DataCleaner()
        clean_data = cleaner.run()
        
        self.results['clean_data'] = clean_data
        self.logger.info(f"Stage 1 completed. Clean data shape: {clean_data.shape}")
        
    def stage_2_feature_engineering(self):
        """Stage 2: Feature Engineering"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: FEATURE ENGINEERING")
        self.logger.info("=" * 60)
        
        builder = FeatureBuilder()
        train_features, test_features = builder.run(self.results['clean_data'])
        
        self.results['train_features'] = train_features
        self.results['test_features'] = test_features
        self.logger.info(f"Stage 2 completed. Train: {train_features.shape}, Test: {test_features.shape}")
        
    def stage_3_demand_modeling(self):
        """Stage 3: Demand Modeling"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: DEMAND MODELING")
        self.logger.info("=" * 60)
        
        modeler = DemandModeler()
        model_results = modeler.run(self.results['train_features'], self.results['test_features'])
        
        self.results['model_results'] = model_results
        self.logger.info(f"Stage 3 completed. Models trained for {len(model_results)} categories")
        
    def stage_4_backtest_evaluation(self):
        """Stage 4: Backtest and Stability Assessment"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: BACKTEST AND STABILITY ASSESSMENT")
        self.logger.info("=" * 60)
        
        evaluator = BacktestEvaluator()
        splits_results, stability_results = evaluator.run(self.results['train_features'])
        
        self.results['splits_results'] = splits_results
        self.results['stability_results'] = stability_results
        self.logger.info(f"Stage 4 completed. Backtest conducted for {stability_results['category'].nunique()} categories")
        
    def stage_5_demand_forecasting(self):
        """Stage 5: Future Demand and Cost Estimation"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: FUTURE DEMAND AND COST ESTIMATION")
        self.logger.info("=" * 60)
        
        forecaster = DemandForecaster()
        forecast_results, item_predictions, uncertainty = forecaster.run(
            self.results['clean_data'], 
            self.results['model_results']
        )
        
        self.results['forecast_results'] = forecast_results
        self.results['item_predictions'] = item_predictions
        self.results['uncertainty'] = uncertainty
        self.logger.info(f"Stage 5 completed. Forecasts generated for {len(forecast_results)} category-day combinations")
        
    def stage_6_pricing_strategy(self):
        """Stage 6: Heuristic Pricing and Replenishment"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: HEURISTIC PRICING AND REPLENISHMENT")
        self.logger.info("=" * 60)
        
        # Extract price elasticities from model results
        price_elasticities = {}
        for category, result in self.results['model_results'].items():
            price_elasticities[category] = result.get('price_elasticity', np.nan)
        
        strategy = PricingReplenishmentStrategy()
        pricing_results = strategy.run(
            self.results['forecast_results'],
            self.results['clean_data'],
            price_elasticities
        )
        
        self.results['pricing_results'] = pricing_results
        self.logger.info(f"Stage 6 completed. Strategy generated for {len(pricing_results)} category-day combinations")
        
    def stage_7_report_generation(self):
        """Stage 7: Results Summary and Export"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 7: RESULTS SUMMARY AND EXPORT")
        self.logger.info("=" * 60)
        
        # Load model results from CSV for report
        model_results_df = pd.read_csv(Config.get_output_path(Config.MODEL_RESULTS_FILE))
        
        generator = ReportGenerator()
        report_results = generator.run(
            self.results['pricing_results'],
            model_results_df,
            self.results['stability_results']
        )
        
        self.results['report_results'] = report_results
        self.logger.info(f"Stage 7 completed. Report saved to: {report_results['report_path']}")
        
    def print_pipeline_summary(self):
        """Print final pipeline summary"""
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        
        summary_info = f"""
Pipeline Execution Completed Successfully!

Key Results:
├── Clean Data Records: {len(self.results['clean_data']):,}
├── Training Features: {self.results['train_features'].shape[0]:,} rows × {self.results['train_features'].shape[1]} features
├── Categories Modeled: {len(self.results['model_results'])}
├── Forecast Combinations: {len(self.results['forecast_results'])}
├── Strategy Combinations: {len(self.results['pricing_results'])}
└── Reports Generated: {len(self.results['report_results']['plot_paths'])} visualizations + analysis report

Output Files:
├── {Config.CLEAN_ITEMS_FILE}
├── {Config.TRAIN_FEATURES_FILE}
├── {Config.TEST_FEATURES_FILE}
├── {Config.MODEL_RESULTS_FILE}
├── {Config.BACKTEST_SPLITS_FILE}
├── {Config.BACKTEST_STABILITY_FILE}
├── {Config.PRICING_REPLENISHMENT_FILE}
└── {Config.ANALYSIS_REPORT_FILE}

Key Metrics:
├── Average Model R²: {pd.read_csv(Config.get_output_path(Config.MODEL_RESULTS_FILE))['test_r2'].mean():.3f}
├── Total Expected Revenue: ¥{self.results['pricing_results']['expected_revenue'].sum():,.2f}
├── Total Expected Profit: ¥{self.results['pricing_results']['expected_profit'].sum():,.2f}
└── Average Markup Rate: {self.results['pricing_results']['markup'].mean():.1%}

All output files are saved in: {Config.OUTPUT_DIR}
Logs are saved in: {Config.LOGS_DIR}
        """
        
        self.logger.info(summary_info)
        print(summary_info)
        
    def run_complete_pipeline(self):
        """Execute the complete 7-stage pipeline"""
        start_time = datetime.now()
        
        self.logger.info("Starting Vegetable Pricing and Replenishment Pipeline")
        self.logger.info(f"Start time: {start_time}")
        
        try:
            # Setup
            self.setup_output_directories()
            
            # Execute all stages
            self.stage_1_data_cleaning()
            self.stage_2_feature_engineering()
            self.stage_3_demand_modeling()
            self.stage_4_backtest_evaluation()
            self.stage_5_demand_forecasting()
            self.stage_6_pricing_strategy()
            self.stage_7_report_generation()
            
            # Summary
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            self.logger.info(f"Pipeline completed successfully!")
            self.logger.info(f"End time: {end_time}")
            self.logger.info(f"Total execution time: {execution_time}")
            
            self.print_pipeline_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            raise e

def main():
    """Main entry point"""
    pipeline = VegetablePricingPipeline()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()