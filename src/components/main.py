import polars as pl
import logging
import sys
import os

# 1. Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.components.validation import ExperimentValidator

# 2. Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    PARQUET_FILE = "data/criteo_uplift.parquet" 
    
    # The split ratio you aimed for (e.g. 0.85 for 85% treatment)
    EXPECTED_RATIO = 0.85 

    if not os.path.exists(PARQUET_FILE):
        logger.error(f"❌ Could not find file: {PARQUET_FILE}")
        return

    logger.info(f"🚀 Loading real dataset from {PARQUET_FILE}...")
    
    # Polars reads Parquet extremely fast
    try:
        df = pl.read_parquet(PARQUET_FILE)
        logger.info(f"✅ Data loaded. Shape: {df.height:,} rows x {df.width} columns")
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        return

    # Basic Sanity Check
    if 'treatment' not in df.columns:
        logger.error("❌ 'treatment' column not found!")
        logger.warning("You must have a column named 'treatment' (0/1) to run validation.")
        return

    # Initialize Validator
    validator = ExperimentValidator(df, treatment_col='treatment', expected_ratio=EXPECTED_RATIO)

    # CHECK 1: Sample Ratio Mismatch (SRM)
    logger.info("--- 1. Checking Sample Ratio Mismatch (SRM) ---")
    srm_result = validator.check_srm()
    
    if srm_result['valid']:
        logger.info(f"SRM Check Passed (p-value={srm_result['p_value']:.4f})")
    else:
        logger.error(f"SRM Check FAILED (p-value={srm_result['p_value']:.4f})")
        logger.info(f"   Observed Counts: {srm_result['observed_counts']}")
        logger.info(f"   Target Ratio: {EXPECTED_RATIO}")

    # CHECK 2: Covariate Balance (SMD)
    logger.info("2. Checking Covariate Balance")
    smd_df = validator.check_covariate_balance()
    
    # Filter for imbalanced features
    imbalanced = smd_df.filter(pl.col("is_balanced") == False)
    
    if imbalanced.height == 0:
        logger.info("All features are balanced (SMD < 0.1)")
    else:
        logger.warning(f"Found {imbalanced.height} imbalanced features:")
        # Show top 5 worst offenders
        print(imbalanced.sort("smd", descending=True).head(5))

    logger.info("Validation complete.")

if __name__ == "__main__":
    main()