import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SegmentAnalyzer:
    """Analyses Uplift predictions to discover actionable segments. Uses Surrogate Modeling (Decision Trees) to explain CATE"""

    def __init__(self, df: pl.DataFrame, feature_cols: List[str]):
        self.df = df
        self.feature_cols = feature_cols
    
    def get_segment_profiles(self, uplift_scores: np.ndarray, top_percentile: float = 0.10) -> pd.DataFrame:
        """Compares the features of the 'High Uplift' segment vs the Rest"""
        logger.info("Generating segment profiles...")

        # 1. Define High Uplift Threshold
        threshold = np.percentile(uplift_scores, 100 * (1-top_percentile))

        # 2. Add scores to df
        temp_df = self.df.select(self.feature_cols).with_columns(
            pl.Series(name="uplift_score", values=uplift_scores)
        )
        
        temp_df = temp_df.with_columns(
            (pl.col("uplift_score") >= threshold).alias("is_persuadable")
        )

        # 3. Aggregate means for Persuadables vs Others
        # Group by boolean, compute mean of all features
        stats = temp_df.group_by("is_persuadable").agg(
            [pl.col(f).mean().alias(f) for f in self.feature_cols]
        ).sort("is_persuadable", descending=True)

        # Convert to pandas for display/diff calculation
        profile_df = stats.to_pandas().set_index("is_persuadable").T

        profile_df.columns = ["Persuadables (Target)", "Others"]

        # Math Fix: Using .abs() on the denominator to handle negative baselines correctly
        profile_df["Diff %"] = ((profile_df["Persuadables (Target)"] - profile_df["Others"]) / profile_df["Others"].abs()) * 100
                
        # Sort by magnitude of difference to find key drivers
        profile_df["Abs Diff"] = profile_df["Diff %"].abs()
        return profile_df.sort_values("Abs Diff", ascending=False).drop(columns=["Abs Diff"])
    
    def explain_with_surrogate(self, uplift_scores: np.ndarray, max_depth: int = 3, min_samples_leaf: int = 50) -> str:
        """Trains a simple Decision Tree Regressor to approx. the complex T-Learner"""
        logger.info("Training surrogate model to explain segments...")
        
        X = self.df.select(self.feature_cols).to_numpy()
        y = uplift_scores
        
        tree_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        tree_model.fit(X, y)
        
        # Extract Rules
        rules = export_text(tree_model, feature_names=self.feature_cols)
        
        self.tree_model = tree_model 
        
        return rules

    def plot_feature_importance(self, save_path: str):
        """Plots feature importance from the surrogate tree."""
        if not hasattr(self, 'tree_model'):
            logger.warning("No surrogate model trained. Run explain_with_surrogate() first.")
            return

        importances = self.tree_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_names_sorted = [self.feature_cols[i] for i in indices]
        
        plt.figure(figsize=(10, 6))
        
        sns.barplot(
            x=importances[indices], 
            y=feature_names_sorted, 
            hue=feature_names_sorted, 
            palette="viridis", 
            legend=False
        )
        
        plt.title("Key Features Driving Uplift (Surrogate Model Importance)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()