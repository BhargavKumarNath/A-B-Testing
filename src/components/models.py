import polars as pl
import numpy as np
import lightgbm as lgb
import pickle
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TLearner:
    """
    T-Learner for Causal Inference.
    Trains two separate models:
    1. Mu_1(x) = E[Y | X, T=1] (Response in Treatment)
    2. Mu_0(x) = E[Y | X, T=0] (Response in Control)
    
    CATE(x) = Mu_1(x) - Mu_0(x)
    """
    
    def __init__(self, features: List[str], learning_rate: float = 0.05, n_estimators: int = 500, seed: int = 500):
        self.features = features
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': learning_rate,
            'num_iterations': n_estimators,
            'num_leaves': 31,
            'verbosity': -1,
            'num_threads': -1,
            'seed': seed
        }

        self.m1 = None
        self.m0 = None
        
    def fit(self, df: pl.DataFrame, treatment_col: str, target_col: str):
        logger.info("Training T-Learner (Two-Model Approach)...")

        df_0 = df.filter(pl.col(treatment_col) == 0)
        df_1 = df.filter(pl.col(treatment_col) == 1)

        X_0 = df_0.select(self.features).to_numpy()
        y_0 = df_0[target_col].to_numpy()

        X_1 = df_1.select(self.features).to_numpy()
        y_1 = df_1[target_col].to_numpy()

        # Train as Booster (native LightGBM)
        train_data_0 = lgb.Dataset(X_0, label=y_0, feature_name=self.features)
        train_data_1 = lgb.Dataset(X_1, label=y_1, feature_name=self.features)

        self.m0 = lgb.train(self.params, train_data_0)
        self.m1 = lgb.train(self.params, train_data_1)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.m0 is None or self.m1 is None:
            raise ValueError("Models not trained. Call fit() first.")

        X = df.select(self.features).to_numpy()

        # Booster API → no warnings
        pred_0 = self.m0.predict(X)
        pred_1 = self.m1.predict(X)

        return pred_1 - pred_0

    def save(self, path: str):
        """Saves the trained models."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'m0': self.m0, 'm1': self.m1, 'features': self.features}, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Loads trained models."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.m0 = data['m0']
            self.m1 = data['m1']
            self.features = data['features']
        logger.info(f"Model loaded from {path}")

class XLearner:
    """X-Learner: The State of the Art Meta Learner for CATE.
    Best for imbalanced treatment groups"""

    def __init__(self, features: List[str], learning_rate: float = 0.05, n_estimators: int = 500):
        self.features = features
        self.clf_params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': learning_rate,
            'num_iterations': n_estimators,
            'verbose': -1,
            'n_jobs': -1
        }
        # Params for the Effect Models (Regression of Residuals)
        self.reg_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': learning_rate,
            'num_iterations': n_estimators,
            'verbose': -1,
            'n_jobs': -1
        }
        self.m0 = None
        self.m1 = None
        self.tau0 = None
        self.tau1 = None
        self.propensity = None

    def fit(self, df: pl.DataFrame, treatment_col: str, target_col: str):
        logger.info("Training X-Learner (Complex Meta-Learner)...")

        # 1. Train Propensity Model (P(T=1|X))
        logger.info("Step 1: Training Propensity Model...")
        X_all = df.select(self.features).to_numpy()
        t_all = df[treatment_col].to_numpy()

        prop_data = lgb.Dataset(X_all, label=t_all, feature_name=self.features)
        self.propensity = lgb.train(self.clf_params, prop_data)
        
        # 2. Train Outcome Models (M0, M1)
        logger.info("Step 2: Training Outcome Models...")
        df_0 = df.filter(pl.col(treatment_col) == 0)
        df_1 = df.filter(pl.col(treatment_col) == 1)
        
        X_0 = df_0.select(self.features).to_numpy()
        y_0 = df_0[target_col].to_numpy()
        X_1 = df_1.select(self.features).to_numpy()
        y_1 = df_1[target_col].to_numpy()
        
        d0 = lgb.Dataset(X_0, label=y_0, feature_name=self.features)
        d1 = lgb.Dataset(X_1, label=y_1, feature_name=self.features)
        
        self.m0 = lgb.train(self.clf_params, d0)
        self.m1 = lgb.train(self.clf_params, d1)
        
        # 3. Calculate Imputed Treatment Effects (D)
        # D1 = Y1 - M0(X1)
        # D0 = M1(X0) - Y0
        logger.info("Step 3: Calculating Imputed Effects...")
        
        # Predict M0 on Treatment Group
        pred_m0_on_1 = self.m0.predict(X_1)
        D_1 = y_1 - pred_m0_on_1
        
        # Predict M1 on Control Group
        pred_m1_on_0 = self.m1.predict(X_0)
        D_0 = pred_m1_on_0 - y_0
        
        # 4. Train Effect Models (Tau0, Tau1)
        # Tau1 predicts D1 using X1
        # Tau0 predicts D0 using X0
        logger.info("Step 4: Training Second-Stage Effect Models...")
        
        d_tau1 = lgb.Dataset(X_1, label=D_1, feature_name=self.features)
        d_tau0 = lgb.Dataset(X_0, label=D_0, feature_name=self.features)
        
        self.tau1 = lgb.train(self.reg_params, d_tau1)
        self.tau0 = lgb.train(self.reg_params, d_tau0)
        
        logger.info("X-Learner Training Complete.")

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.tau0 is None:
            raise ValueError("Model not trained.")
            
        X = df.select(self.features).to_numpy()
        
        # Get Propensities g(x)
        g = self.propensity.predict(X)
        
        # Get Estimated Effects
        # Tau0 predicts effect for Control units
        # Tau1 predicts effect for Treated units
        t0_pred = self.tau0.predict(X)
        t1_pred = self.tau1.predict(X)
        
        # Weighting Formula:
        # CATE = g(x)*t0(x) + (1-g(x))*t1(x)
        cate = (g * t0_pred) + ((1 - g) * t1_pred)
        
        return cate
