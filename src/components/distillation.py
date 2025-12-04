import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import logging
import pickle
from typing import List, Any, Dict
logger = logging.getLogger(__name__)

class DistillationEngine:
    """Compresses a complex Casual Model (Teacher) into a lightweight, interpretable model (Student) for low latency production inference"""

    def __init__(self, teacher_model: Any, feature_cols: List[str], student_type: str = 'tree'):
        self.teacher = teacher_model
        self.features = feature_cols
        self.student = None
        self.student_type = student_type

    def train_student(self, df: pl.DataFrame, max_depth: int = 4):
        """Trains the student to mimic the teacher's CATE predictions"""
        logger.info(f"Starting Knowledge Distillation (Student: {self.student_type})...")

        # 1. Generate Soft Labels (Teacher Predictions)
        # Student learns the "smooth" logic of the teacher, not the noisy raw data
        logger.info("Generating teacher predictions (Soft Labels)...")
        cate_preds = self.teacher.predict(df)

        X = df.select(self.features).to_numpy()
        y = cate_preds

        # 2. Initialise Student
        if self.student_type == 'tree':
            self.student = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50)
        elif self.student_type == 'linear':
            self.student = LinearRegression()
        else:
            raise ValueError("Unsupported student type. Use 'tree' or 'linear'.")

        # 3. Fit Student
        logger.info(f"Fitting student model on {len(X)} samples...")
        self.student.fit(X, y)

        # 4. Evaluate Fidelity
        student_preds = self.student.predict(X)
        r2 = r2_score(y, student_preds)
        rmse = np.sqrt(mean_squared_error(y, student_preds))

        logger.info(f"Distillation Complete")
        logger.info(f"Student Fidelity (R2): {r2:.4f} (Higher is better)")
        logger.info(f"Student RMSE: {rmse:.6f} (Lower is better)")

        return r2
    
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Fast inference using the student model."""
        if self.student is None:
            raise ValueError("Student not trained")
        X = df.select(self.features).to_numpy()

        return self.student.predict(X)
    
    def save_student(self, path: str):
        """Saves only the lightweight student model"""
        with open(path, 'wb') as f:
            pickle.dump(self.student, f)
        logger.info(f"Student model saved to {path}")
