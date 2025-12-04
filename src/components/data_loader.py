import polars as pl
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles efficient loading and memory optimization of Criteo datasets using Polars"""
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.required_columns = [
            'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
            'treatment', 'conversion', 'visit', 'exposure'
        ]
            
    def _optimize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Downcasts types to reduce memory usage
        float64 -> float32
        int64 (binary) -> int8"""
        logger.info("Optimizing data types for memory efficiency...")

        # Define casting rules
        cast_exprs = []
        for col in df.columns:
            dtype = df.schema[col]

            if dtype == pl.Float64:
                cast_exprs.append(pl.col(col).cast(pl.Float32))
            
            elif col in ['treatment', 'conversion', 'visit', 'exposure']:
                cast_exprs.append(pl.col(col).cast(pl.Int8))

        return df.with_columns(cast_exprs)
    
    def load(self) -> pl.DataFrame:
        """Loads the parquet file, verifies schema, and optimizes memory"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        logger.info(f"Loading data from {self.filepath}...")

        try:
            df = pl.read_parquet(self.filepath)
        except Exception as e:
            logger.error(f"Failed to read parquet file: {e}")
            raise

        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        initial_mem = df.estimated_size("mb")
        logger.info(f"Initial memory usage: {initial_mem:.2f} MB")

        # Optimisation
        df_optimized = self._optimize_types(df)
        final_mem = df_optimized.estimated_size("mb")
        logger.info(f"Optimized memory usage: {final_mem:.2f} MB")
        logger.info(f"Memory saved: {initial_mem - final_mem:.2f} MB")
        logger.info(f"Loaded {df_optimized.height} rows and {df_optimized.width} columns.")

        return df_optimized

if __name__ == "__main__":
    pass
        