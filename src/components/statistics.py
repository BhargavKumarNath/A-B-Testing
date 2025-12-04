import polars as pl
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    metric: str
    variant: str
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    std_error: float

class FrequentistEngine:
    """
    Handles classical frequentist A/B testing calculations, 
    including T-tests and Variance Reduction (CUPED).
    """
    
    def __init__(self, df: pl.DataFrame, treatment_col: str = 'treatment'):
        self.df = df
        self.treatment_col = treatment_col
        self.feature_cols = [c for c in df.columns if c.startswith('f')]

    def calculate_ate(self, target_col: str, alpha: float = 0.05) -> TestResult:
        """
        Calculates Average Treatment Effect using Welch's t-test (unequal variances).
        """
        logger.info(f"Calculating ATE for {target_col}...")
        
        # 1. Aggregations using Polars (Instant)
        # We compute Count, Mean, Variance for Treatment (1) and Control (0)
        agg = self.df.group_by(self.treatment_col).agg([
            pl.count(target_col).alias("n"),
            pl.mean(target_col).alias("mean"),
            pl.var(target_col).alias("var")
        ]).sort(self.treatment_col)
        
        # Extract values
        # Row 0 is Control, Row 1 is Treatment (due to sort)
        n_c = agg["n"][0]
        mu_c = agg["mean"][0]
        var_c = agg["var"][0]
        
        n_t = agg["n"][1]
        mu_t = agg["mean"][1]
        var_t = agg["var"][1]
        
        # 2. Statistics
        effect = mu_t - mu_c
        relative_effect = effect / mu_c if mu_c != 0 else 0.0
        
        # Standard Error (Welch's)
        se = np.sqrt((var_c / n_c) + (var_t / n_t))
        
        # Z-score (N is large, t-dist converges to normal)
        z_score = stats.norm.ppf(1 - alpha / 2)
        ci_lower = effect - z_score * se
        ci_upper = effect + z_score * se
        
        # P-value (two-sided)
        t_stat = effect / se
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return TestResult(
            metric=target_col,
            variant="Simple Difference",
            control_mean=mu_c,
            treatment_mean=mu_t,
            absolute_effect=effect,
            relative_effect=relative_effect,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_val,
            std_error=se
        )

    def calculate_cuped(self, target_col: str, covariates: Optional[List[str]] = None) -> TestResult:
        """
        Calculates Variance-Reduced ATE using CUPED (Regression Adjustment).
        Y_cuped = Y - theta * (X - mean(X))
        """
        if covariates is None:
            covariates = self.feature_cols
            
        logger.info(f"Calculating CUPED ATE for {target_col} using {len(covariates)} covariates...")
        
        # We need to calculate theta for the covariates.
        # For simplicity and speed in a vectorized way, we'll use a single dominant covariate 
        # or the sum of key features if multiple are passed, 
        # OR we perform the full matrix adjustment.
        
        # FANG approach: Use a linear combination.
        # Y_cuped = Y - (X dot Beta).
        # We estimate Beta using the total dataset (pooled).
        
        # 1. Prepare Matrices
        # We'll use a subset of rows to estimate Beta if memory is tight, 
        # but 14M rows * 12 floats is ~1.5GB, fitting in RAM.
        
        try:
            # We add a constant column for intercept
            X = self.df.select(covariates).to_numpy()
            Y = self.df.select(target_col).to_numpy()
            
            # Center X to avoid intercept issues (Standard CUPED)
            X_mean = np.mean(X, axis=0)
            X_centered = X - X_mean
            
            # Estimate Theta (Beta) using OLS: (X'X)^-1 X'Y
            # Using simple Covariance/Variance ratio for multivariate:
            # Theta = Cov(Y, X) * Var(X)^-1
            
            # Using numpy least squares is safer and handles multivariate
            theta, _, _, _ = np.linalg.lstsq(X_centered, Y, rcond=None)
            
            # Calculate CUPED Metric
            # Y_adj = Y - (X - X_mean) * theta
            adjustment = np.dot(X_centered, theta)
            y_cuped_data = Y.flatten() - adjustment.flatten()
            
            # Now we have a new "CUPED Metric". We create a temporary DataFrame to run the t-test again.
            # This is lightweight.
            temp_df = self.df.select(pl.col(self.treatment_col)).with_columns(
                pl.Series(values=y_cuped_data).alias(f"{target_col}_cuped")
            )
            
            # Reuse the calculate_ate method on this new metric
            # Here we just test the difference.
            
            cuped_engine = FrequentistEngine(temp_df, self.treatment_col)
            result = cuped_engine.calculate_ate(f"{target_col}_cuped")
            result.variant = "CUPED Adjusted"
                        
            return result
            
        except Exception as e:
            logger.error(f"CUPED calculation failed: {e}")
            raise
