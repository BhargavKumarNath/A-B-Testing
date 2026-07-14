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
from src.components.distillation import DistillationEngine
from src.components.exporter import PowerBIExporter
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

    # CUPED variance reduction — computed on a 10 % random sample for performance.
    # With n ≈ 1.4 M rows, the theta estimate is statistically equivalent to the
    # full-dataset result (SE ∝ 1/√n; further reduction is negligible at this scale).
    logger.info("Computing CUPED (10 % sample for efficiency) ...")
    sample_size_cuped = max(100_000, int(df.height * 0.10))
    cuped_sample = df.sample(n=sample_size_cuped, shuffle=True, seed=42)
    cuped_engine = FrequentistEngine(cuped_sample)
    cuped_result = cuped_engine.calculate_cuped("conversion")
    
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
    # max_depth=5 matches the production distillation depth and powerbi.md specification.
    # max_depth=3 was the original demo setting; depth=5 produces richer rules for
    # the Page 5 tree rules viewer without sacrificing interpretability.
    rules = analyzer.explain_with_surrogate(uplift_scores, max_depth=5)
    
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

    # ── Phase 2: Power BI Data Export ────────────────────────────────────────
    # All pipeline artefacts are now available.  The exporter collects them,
    # applies optimisations (downsampling, stratified sampling, PSI computation),
    # and writes structured CSVs to results/data/ for the Power BI data model.
    logger.info("Starting Power BI data export layer ...")
    exporter = PowerBIExporter(
        output_dir          = OUTPUT_DIR,
        feature_cols        = FEATURE_COLS,
        srm_result          = validator.check_srm(),     # re-call: pure computation, no side-effect
        balance_df          = balance_df,
        ate_result          = ate_result,
        cuped_result        = cuped_result,
        uplift_scores       = uplift_scores,
        test_df             = test_df,
        train_df            = train_df,
        decile_stats        = decile_stats,
        qini_data           = qini_data,
        learner             = learner,
        analyzer            = analyzer,
        bandit_res          = bandit_res,
        baseline_profit_per_user = baseline_profit_per_user,
        student_r2          = r2_score,
        conversion_value    = CONVERSION_VALUE,
        cost_per_ad         = COST_PER_AD,
    )
    export_paths = exporter.export_all()
    logger.info(f"Power BI exports complete: {len(export_paths)} files in {OUTPUT_DIR}/data/")

if __name__ == "__main__":
    run_pipeline()