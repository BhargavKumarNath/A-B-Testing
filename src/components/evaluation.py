import polars as pl
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class UpliftEvaluator:
    """
    Evaluates Uplift Model performance using:
    1. Uplift by Decile (Verifying the rank ordering of effects)
    2. Qini Curve (Cumulative gain in conversions)
    3. Bootstrapped Confidence Intervals (Statistical Rigor)
    """

    def __init__(self, df: pl.DataFrame, treatment_col: str = 'treatment', target_col: str = 'conversion'):
        self.df = df
        self.treatment_col = treatment_col
        self.target_col = target_col

    def get_decile_stats(self, pred_uplift: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
        """
        Groups users into bins based on predicted uplift and calculates
        the ACTUAL Average Treatment Effect (ATE) within each bin.
        """
        eval_df = self.df.select([self.treatment_col, self.target_col]).with_columns(
            pl.Series(name="pred_uplift", values=pred_uplift)
        )

        eval_df = eval_df.with_columns(
            pl.col("pred_uplift").qcut(n_bins, labels=[str(i) for i in range(n_bins)], allow_duplicates=True).cast(pl.String).alias("bin")
        )

        stats = eval_df.group_by("bin").agg([
            pl.count().alias("n_obs"),
            pl.col("pred_uplift").mean().alias("mean_pred_uplift"),
            pl.col(self.target_col).filter(pl.col(self.treatment_col) == 0).mean().alias("mean_y_c"),
            pl.col(self.target_col).filter(pl.col(self.treatment_col) == 0).count().alias("n_c"),
            pl.col(self.target_col).filter(pl.col(self.treatment_col) == 1).mean().alias("mean_y_t"),
            pl.col(self.target_col).filter(pl.col(self.treatment_col) == 1).count().alias("n_t"),
        ]).sort("bin", descending=True)

        stats = stats.with_columns(
            (pl.col("mean_y_t") - pl.col("mean_y_c")).alias("actual_lift")
        )
        return stats

    def _calculate_qini_curve(self, t: np.ndarray, y: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Helper to calculate raw Qini curve points."""
        # Sort indices by score descending
        sorted_indices = np.argsort(scores)[::-1]
        t_sorted = t[sorted_indices]
        y_sorted = y[sorted_indices]

        # Cumulative sums
        cum_y_t = np.cumsum(y_sorted * (t_sorted == 1))
        cum_n_t = np.cumsum(t_sorted == 1)
        cum_y_c = np.cumsum(y_sorted * (t_sorted == 0))
        cum_n_c = np.cumsum(t_sorted == 0)

        # Avoid division by zero
        cum_n_t[cum_n_t == 0] = 1
        cum_n_c[cum_n_c == 0] = 1

        # Incremental Gains (Uplift definition)
        # Curve = (Yt/Nt - Yc/Nc) * (Nt + Nc)
        conversion_rate_t = cum_y_t / cum_n_t
        conversion_rate_c = cum_y_c / cum_n_c
        curve_y = (conversion_rate_t - conversion_rate_c) * (cum_n_t + cum_n_c)
        
        # X-axis (Fraction of population)
        curve_x = np.linspace(0, 1, len(curve_y))
        
        # Area Under Curve (Trapezoidal)
        auuc = np.trapz(curve_y, curve_x)
        
        return curve_x, curve_y, auuc

    def get_qini_data(self, pred_uplift: np.ndarray) -> Dict:
        """Standard Qini calculation."""
        t = self.df[self.treatment_col].to_numpy()
        y = self.df[self.target_col].to_numpy()
        x, y_curve, auuc = self._calculate_qini_curve(t, y, pred_uplift)
        
        # Downsample for plotting (1000 points)
        indices = np.linspace(0, len(x) - 1, 1000).astype(int)
        
        return {
            "x": x[indices],
            "y": y_curve[indices],
            "auuc": auuc
        }

    def get_bootstrapped_qini(self, pred_uplift: np.ndarray, n_bootstraps: int = 50) -> Dict:
        """
        Calculates Qini curve with Confidence Intervals using Bootstrapping.
        Resamples the dataset N times to estimate uncertainty.
        """
        logger.info(f"Bootstrapping Qini Curve ({n_bootstraps} iterations)...")
        
        t_full = self.df[self.treatment_col].to_numpy()
        y_full = self.df[self.target_col].to_numpy()
        scores_full = pred_uplift
        
        n_samples = len(y_full)
        
        # Common X-axis for interpolation (0% to 100%)
        common_x = np.linspace(0, 1, 1000)
        bootstrapped_curves = []
        auucs = []
        
        for i in range(n_bootstraps):
            # Resample indices with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            
            t_b = t_full[indices]
            y_b = y_full[indices]
            s_b = scores_full[indices]
            
            # Calculate curve
            x_b, y_b_curve, auuc_b = self._calculate_qini_curve(t_b, y_b, s_b)
            
            # Interpolate to common x-axis to average them
            # We interpolate y_b_curve onto common_x based on x_b
            interp_y = np.interp(common_x, x_b, y_b_curve)
            
            bootstrapped_curves.append(interp_y)
            auucs.append(auuc_b)
            
        bootstrapped_curves = np.array(bootstrapped_curves)
        
        # Calculate statistics
        mean_curve = np.mean(bootstrapped_curves, axis=0)
        lower_bound = np.percentile(bootstrapped_curves, 2.5, axis=0) # 95% CI Lower
        upper_bound = np.percentile(bootstrapped_curves, 97.5, axis=0) # 95% CI Upper
        
        mean_auuc = np.mean(auucs)
        std_auuc = np.std(auucs)
        
        return {
            "x": common_x,
            "y_mean": mean_curve,
            "y_lower": lower_bound,
            "y_upper": upper_bound,
            "auuc_mean": mean_auuc,
            "auuc_std": std_auuc
        }