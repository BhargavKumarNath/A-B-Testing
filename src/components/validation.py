import polars as pl
import numpy as np
from scipy.stats import chisquare
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class ExperimentValidator:
    """
    Performs integrity checks on the experimental data:
    1. Sample Ratio Mismatch (SRM)
    2. Covariate Balance (Standardized Mean Differences)
    """
    
    def __init__(self, df: pl.DataFrame, treatment_col: str = 'treatment', expected_ratio: float = 0.85):
        self.df = df
        self.treatment_col = treatment_col
        self.expected_ratio = expected_ratio
        self.feature_cols = [c for c in df.columns if c.startswith('f')]

    def check_srm(self, alpha: float = 0.001) -> Dict:
        """
        Performs a Chi-Square test to detect Sample Ratio Mismatch.
        Using a strict alpha (0.001) because with 14M rows, even tiny deviations can trigger p < 0.05.
        """
        logger.info("Checking for Sample Ratio Mismatch (SRM)...")
        
        counts = self.df[self.treatment_col].value_counts().sort(self.treatment_col)
        
        # Extract counts (assuming 0 is Control, 1 is Treatment)
        # Polars value_counts returns struct or df, we extract rows.
        # We need to ensure we map 0 and 1 correctly regardless of sort order
        n_obs = self.df.height
        
        # Get actual counts
        # We filter explicitly to be safe against missing keys if a group is empty (unlikely)
        n_control = self.df.filter(pl.col(self.treatment_col) == 0).height
        n_treat = self.df.filter(pl.col(self.treatment_col) == 1).height
        
        observed = [n_control, n_treat]
        
        # Expected counts based on ratio
        # Control ratio = 1 - expected_ratio
        expected = [n_obs * (1 - self.expected_ratio), n_obs * self.expected_ratio]
        
        # Chi-Square Test
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        
        is_valid = bool(p_value > alpha)
        
        result = {
            "test": "SRM Chi-Square",
            "observed_counts": {"control": n_control, "treatment": n_treat},
            "expected_ratio": self.expected_ratio,
            "p_value": p_value,
            "valid": is_valid
        }
        
        if not is_valid:
            logger.warning(f"SRM DETECTED! p-value: {p_value:.10f}. Observed ratio: {n_treat/n_obs:.4f} vs Expected: {self.expected_ratio}")
        else:
            logger.info(f"SRM Check Passed. p-value: {p_value:.4f}")
            
        return result

    def check_covariate_balance(self) -> pl.DataFrame:
        """
        Calculates Standardized Mean Difference (SMD) for all features.
        SMD = (Mean_T - Mean_C) / sqrt((Var_T + Var_C) / 2)
        Target: |SMD| < 0.1
        """
        logger.info("Checking covariate balance (SMD)...")
        
        # Aggregations for mean and variance
        agg_exprs = []
        for col in self.feature_cols:
            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            agg_exprs.append(pl.col(col).var().alias(f"{col}_var"))
            
        stats = self.df.group_by(self.treatment_col).agg(agg_exprs)
        
        # Extract stats for T (1) and C (0)
        # We'll pivot or just filter. Filtering is clearer in code.
        stats_t = stats.filter(pl.col(self.treatment_col) == 1)
        stats_c = stats.filter(pl.col(self.treatment_col) == 0)
        
        smd_data = []
        
        for col in self.feature_cols:
            mean_t = stats_t[f"{col}_mean"][0]
            mean_c = stats_c[f"{col}_mean"][0]
            var_t = stats_t[f"{col}_var"][0]
            var_c = stats_c[f"{col}_var"][0]
            
            # SMD Formula
            pooled_std = np.sqrt((var_t + var_c) / 2)
            if pooled_std == 0:
                smd = 0.0
            else:
                smd = (mean_t - mean_c) / pooled_std
                
            smd_data.append({
                "feature": col,
                "smd": smd,
                "is_balanced": abs(smd) < 0.1
            })
            
        return pl.DataFrame(smd_data)