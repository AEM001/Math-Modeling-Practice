"""
Main pipeline entry point for single product restocking and pricing optimization.

Usage:
    python src/main.py [options]

Options:
    --date: Target date for optimization (default: 2023-07-01)
    --data-dir: Data directory path (default: ./data)
    --output-dir: Output directory path (default: ./outputs)
    --use-elasticity: Use price elasticity model (default: False)
    --solver: Optimization solver (default: CBC)
    --quick-mode: Use quick baseline forecasting (default: False)
    --no-viz: Skip visualization generation (default: False)
"""
import sys
import os
import argparse
import logging
from datetime import datetime

# 添加src目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from config import *
from io_utils import prepare_training_data, save_results
from forecast import forecast_sales_and_prices, quick_forecast_baseline
from screen import screen_candidates, analyze_screening_results, export_screening_report
from optimize import optimize_restocking
from pricing import integrate_results
from visualize import create_all_visualizations, generate_visualization_report

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Single Product Restocking and Pricing Optimization')
    
    parser.add_argument('--date', type=str, default=TARGET_DATE,
                       help='Target optimization date (YYYY-MM-DD)')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory path')
    
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory path')
    
    parser.add_argument('--use-elasticity', action='store_true',
                       help='Use price elasticity model')
    
    parser.add_argument('--solver', type=str, default=SOLVER_CONFIG['default_solver'],
                       choices=['CBC', 'GLPK'],
                       help='Optimization solver')
    
    parser.add_argument('--quick-mode', action='store_true',
                       help='Use quick baseline forecasting method')
    
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    parser.add_argument('--max-candidates', type=int, default=N_CANDIDATES_MAX,
                       help='Maximum number of candidate products')
    
    parser.add_argument('--min-shelf', type=int, default=MIN_SHELF_COUNT,
                       help='Minimum shelf count')
    
    parser.add_argument('--max-shelf', type=int, default=MAX_SHELF_COUNT,
                       help='Maximum shelf count')
    
    return parser.parse_args()

def validate_arguments(args):
    """
    验证命令行参数
    """
    # 验证日期格式
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {args.date}. Expected YYYY-MM-DD")
    
    # 验证目录存在
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # 验证参数范围
    if args.max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    
    if args.min_shelf <= 0 or args.max_shelf <= 0 or args.min_shelf > args.max_shelf:
        raise ValueError("Invalid shelf count parameters")
    
    logger.info("Arguments validated successfully")

def setup_output_directories(output_dir):
    """
    设置输出目录
    """
    dirs_to_create = [
        output_dir,
        os.path.join(output_dir, 'results'),
        os.path.join(output_dir, 'figs')
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Output directories set up in {output_dir}")

def run_forecasting(df, args):
    """
    运行预测模块
    """
    logger.info("=" * 50)
    logger.info("STEP 2: FORECASTING")
    logger.info("=" * 50)
    
    if args.quick_mode:
        logger.info("Using quick baseline forecasting method...")
        forecast_df = quick_forecast_baseline(df, args.date)
        forecaster = None
    else:
        logger.info("Using machine learning forecasting method...")
        forecast_df, forecaster = forecast_sales_and_prices(df, args.date)
    
    if len(forecast_df) == 0:
        raise RuntimeError("Forecasting failed - no predictions generated")
    
    logger.info(f"Forecasting completed: {len(forecast_df)} products predicted")
    
    # 保存预测结果
    forecast_path = os.path.join(args.output_dir, 'results', 'forecast_results.csv')
    save_results(forecast_df, forecast_path)
    
    return forecast_df, forecaster

def run_screening(forecast_df, historical_df, args):
    """
    运行筛选模块
    """
    logger.info("=" * 50)
    logger.info("STEP 3: CANDIDATE SCREENING")
    logger.info("=" * 50)
    
    screening_config = {
        'min_sales': MIN_DISPLAY_QTY,
        'max_candidates': args.max_candidates,
        'min_quality': QUALITY_THRESHOLDS['min_r2_score'],
        'ensure_diversity': False
    }
    
    candidates_df = screen_candidates(forecast_df, historical_df, screening_config)
    
    if len(candidates_df) == 0:
        raise RuntimeError("Candidate screening failed - no candidates selected")
    
    # 分析筛选结果
    screening_analysis = analyze_screening_results(candidates_df, forecast_df)
    
    # 导出筛选报告
    screening_report_path = os.path.join(args.output_dir, 'results', 'screening_report.txt')
    export_screening_report(candidates_df, screening_analysis, screening_report_path)
    
    # 保存候选产品
    candidates_path = os.path.join(args.output_dir, 'results', 'candidates.csv')
    save_results(candidates_df, candidates_path)
    
    logger.info(f"Screening completed: {len(candidates_df)} candidates selected")
    
    return candidates_df, screening_analysis

def run_optimization(candidates_df, args):
    """
    运行优化模块
    """
    logger.info("=" * 50)
    logger.info("STEP 4: OPTIMIZATION")
    logger.info("=" * 50)
    
    # 更新配置参数
    optimization_config = {
        'use_elasticity': args.use_elasticity,
        'solver': args.solver,
        'time_limit': SOLVER_CONFIG['time_limit'],
        'min_shelf': args.min_shelf,
        'max_shelf': args.max_shelf
    }
    
    # 临时更新全局配置
    global MIN_SHELF_COUNT, MAX_SHELF_COUNT
    MIN_SHELF_COUNT = args.min_shelf
    MAX_SHELF_COUNT = args.max_shelf
    
    solution_df, optimization_summary = optimize_restocking(candidates_df, optimization_config)
    
    if solution_df is None or len(solution_df) == 0:
        raise RuntimeError("Optimization failed - no solution found")
    
    logger.info(f"Optimization completed: {len(solution_df)} products selected")
    
    return solution_df, optimization_summary

def run_result_integration(solution_df, optimization_summary, args):
    """
    运行结果整合模块
    """
    logger.info("=" * 50)
    logger.info("STEP 5: RESULT INTEGRATION")
    logger.info("=" * 50)
    
    final_solution, combined_summary, export_paths = integrate_results(
        solution_df, optimization_summary
    )
    
    logger.info("Result integration completed")
    logger.info(f"Final plan exported to: {export_paths['plan_csv']}")
    
    return final_solution, combined_summary, export_paths

def run_visualization(forecast_df, candidates_df, solution_df, summary, args):
    """
    运行可视化模块
    """
    if args.no_viz:
        logger.info("Skipping visualization as requested")
        return {}
    
    logger.info("=" * 50)
    logger.info("STEP 6: VISUALIZATION")
    logger.info("=" * 50)
    
    try:
        figures = create_all_visualizations(forecast_df, candidates_df, solution_df, summary)
        
        # 生成可视化报告
        report_path = os.path.join(args.output_dir, 'figs', 'visualization_report.txt')
        generate_visualization_report(figures, report_path)
        
        logger.info(f"Visualization completed: {len(figures)} charts generated")
        
        return figures
    
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        return {}

def print_final_summary(summary, export_paths):
    """
    打印最终摘要
    """
    logger.info("=" * 50)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 50)
    
    logger.info("\n📊 FINAL RESULTS SUMMARY:")
    logger.info(f"  Selected Products: {summary.get('selected_count', 0)}")
    logger.info(f"  Total Stock: {summary.get('total_stock_kg', 0):.2f} kg")
    logger.info(f"  Total Revenue: {summary.get('total_revenue', 0):.2f} yuan")
    logger.info(f"  Total Profit: {summary.get('total_profit', 0):.2f} yuan")
    logger.info(f"  Profit Margin: {summary.get('profit_margin', 0):.2%}")
    logger.info(f"  Average Markup: {summary.get('avg_markup', 0):.2%}")
    
    logger.info("\n📁 OUTPUT FILES:")
    for file_type, file_path in export_paths.items():
        logger.info(f"  {file_type}: {file_path}")
    
    logger.info("\n✅ Optimization pipeline completed successfully!")

def main():
    """
    主函数
    """
    start_time = datetime.now()
    
    try:
        # 解析参数
        args = parse_arguments()
        validate_arguments(args)
        
        # 设置输出目录
        setup_output_directories(args.output_dir)
        
        logger.info("=" * 50)
        logger.info("SINGLE PRODUCT RESTOCKING OPTIMIZATION")
        logger.info("=" * 50)
        logger.info(f"Target Date: {args.date}")
        logger.info(f"Data Directory: {args.data_dir}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Use Elasticity: {args.use_elasticity}")
        logger.info(f"Solver: {args.solver}")
        logger.info(f"Quick Mode: {args.quick_mode}")
        
        # STEP 1: 数据准备
        logger.info("=" * 50)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 50)
        
        historical_df, sellable_products = prepare_training_data(args.date)
        
        if len(historical_df) == 0:
            raise RuntimeError("No historical data available")
        
        logger.info(f"Data loaded: {len(historical_df)} records for {len(sellable_products)} products")
        
        # STEP 2: 预测
        forecast_df, forecaster = run_forecasting(historical_df, args)
        
        # STEP 3: 候选筛选
        candidates_df, screening_analysis = run_screening(forecast_df, historical_df, args)
        
        # STEP 4: 优化
        solution_df, optimization_summary = run_optimization(candidates_df, args)
        
        # STEP 5: 结果整合
        final_solution, combined_summary, export_paths = run_result_integration(
            solution_df, optimization_summary, args
        )
        
        # STEP 6: 可视化
        figures = run_visualization(forecast_df, candidates_df, final_solution, 
                                  combined_summary, args)
        
        # 打印最终摘要
        print_final_summary(combined_summary, export_paths)
        
        # 计算总运行时间
        end_time = datetime.now()
        runtime = end_time - start_time
        logger.info(f"\n⏱️  Total runtime: {runtime}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)