import logging
import time
from pathlib import Path
import polars as pl
import numpy as np

# Component Imports
from src.components.data_loader import DataLoader
from src.components.validation import ExperimentValidator
from src.components.statistics import FrequentistEngine
from src.components.models import XLearner 
from src.components.evaluation import UpliftEvaluator
from src.components.segmentation import SegmentAnalyzer
from src.components.bandit import BanditSimulator 
from src.components.distillation import DistillationEngine # <--- NEW
from src.utils.plotting import plot_uplift_by_decile, plot_bootstrapped_qini, plot_bandit_performance

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIG
DATA_PATH = "data/criteo_uplift.parquet"
OUTPUT_DIR = "results"
FEATURE_COLS = [f"f{i}" for i in range(12)]

# Business Params for Profit Bandit
CONVERSION_VALUE = 10.0  # Revenue per conversion
COST_PER_AD = 0.10       # Cost per impression

def run_pipeline():
    start_time = time.time()
    logger.info("Starting Uplift A/B Testing Pipeline...")
    
    # 1. Data Loading
    loader = DataLoader(DATA_PATH)
    df = loader.load()
    
    # 2. Validation
    logger.info("Phase 1: Experiment Validation")
    validator = ExperimentValidator(df)
    if not validator.check_srm()['valid']:
        logger.warning("SRM Detected! Proceeding with caution...")
    
    balance_df = validator.check_covariate_balance()
    logger.info(f"Covariate Balance Checked. Max Imbalance: {balance_df['smd'].abs().max():.4f}")
    
    # 3. Classical A/B Testing
    logger.info("Phase 2: Classical A/B Testing")
    stat_engine = FrequentistEngine(df)
    ate_result = stat_engine.calculate_ate("conversion")
    
    logger.info(f"Global ATE: {ate_result.absolute_effect:.6f} (Lift: {ate_result.relative_effect:.2%})")
    
    # 4. Advanced Uplift Modeling (X-Learner)
    logger.info("Phase 3: Uplift Modeling (X-Learner)")
    
    # Random Split
    df = df.with_columns(pl.Series(name="rand_split", values=np.random.rand(df.height)))
    train_df = df.filter(pl.col("rand_split") < 0.8)
    test_df = df.filter(pl.col("rand_split") >= 0.8)
    
    # Using X-Learner (State of the Art for Imbalanced Data)
    learner = XLearner(features=FEATURE_COLS, n_estimators=100) # 100 trees for speed in demo, use 500 for prod
    learner.fit(train_df, treatment_col="treatment", target_col="conversion")
    
    logger.info("Predicting Uplift on Test Set...")
    uplift_scores = learner.predict(test_df)
    
    # 5. Evaluation with Uncertainty (Bootstrapping)
    logger.info("Phase 4: Evaluation (Bootstrapped)")
    evaluator = UpliftEvaluator(test_df)
    
    decile_stats = evaluator.get_decile_stats(uplift_scores, n_bins=10)
    
    # Bootstrapped Qini (Calculates 95% Confidence Intervals)
    # n_bootstraps=20 for speed in demo, use 100+ for prod
    qini_data = evaluator.get_bootstrapped_qini(uplift_scores, n_bootstraps=20)
    
    plot_uplift_by_decile(decile_stats, f"{OUTPUT_DIR}/uplift_deciles.png")
    plot_bootstrapped_qini(qini_data, f"{OUTPUT_DIR}/qini_curve.png")
    
    # 6. Segmentation & Explainability
    logger.info("Phase 5: Segmentation Analysis")
    analyzer = SegmentAnalyzer(test_df, FEATURE_COLS)
    profile_df = analyzer.get_segment_profiles(uplift_scores, top_percentile=0.15)
    rules = analyzer.explain_with_surrogate(uplift_scores, max_depth=3)
    
    profile_df.to_csv(f"{OUTPUT_DIR}/segment_profile.csv")
    with open(f"{OUTPUT_DIR}/segment_rules.txt", "w") as f:
        f.write(rules)
        
    # 7. Profit-Aware Bandit Simulation
    logger.info("Phase 6: Bandit Simulation (Profit Optimization)")
    # Simulation: Optimizing Net Profit, NOT just CTR
    sim = BanditSimulator(test_df, FEATURE_COLS, "conversion", "treatment")
    
    bandit_res = sim.run_replay(
        sample_size=1_000_000, 
        conversion_value=CONVERSION_VALUE, 
        cost_per_ad=COST_PER_AD
    )
    
    # Calculate Baseline Profit (Fixed Strategy)
    # If we treat everyone: Profit = (GlobalCTR * Value) - Cost
    global_ctr = df["conversion"].mean()
    baseline_profit_per_user = (global_ctr * CONVERSION_VALUE) - (COST_PER_AD * df["treatment"].mean())
    
    logger.info("=== Economic Impact ===")
    logger.info(f"Fixed Strategy Profit/User: ${baseline_profit_per_user:.4f}")
    logger.info(f"Bandit Strategy Profit/User: ${bandit_res['avg_profit']:.4f}")
    
    plot_bandit_performance(bandit_res['history_reward'], baseline_profit_per_user, f"{OUTPUT_DIR}/bandit_profit.png")
    
    # 8. Engineering: Knowledge Distillation
    logger.info("Phase 7: Production Engineering (Distillation)")
    # Compressing the heavy X-Learner into a fast Student model
    distiller = DistillationEngine(teacher_model=learner, feature_cols=FEATURE_COLS, student_type='tree')
    r2_score = distiller.train_student(test_df, max_depth=5)
    
    distiller.save_student(f"{OUTPUT_DIR}/production_uplift_model.pkl")
    logger.info(f"Student Model Fidelity (R2): {r2_score:.4f}")

    logger.info(f"Full Pipeline Complete. Assets saved to {OUTPUT_DIR}/")
    logger.info(f"Total Time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    run_pipeline()