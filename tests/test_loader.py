import sys
import os
import pytest
import polars as pl
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_loader import DataLoader

def create_dummy_parquet(path: Path):
    """Creates a small dummy parquet file matching Criteo structure."""
    data = {
        f"f{i}": np.random.rand(100).astype("float64") for i in range(12)
    }
    data.update({
        "treatment": np.random.randint(0, 2, 100).astype("int64"),
        "conversion": np.random.randint(0, 2, 100).astype("int64"),
        "visit": np.random.randint(0, 2, 100).astype("int64"),
        "exposure": np.random.randint(0, 2, 100).astype("int64"),
    })
    
    df = pl.DataFrame(data)
    df.write_parquet(path)

def test_data_loader_optimization(tmp_path):
    """Test that the loader loads data and downcasts types correctly."""
    
    # 1. Setup
    dummy_file = tmp_path / "criteo_dummy.parquet"
    create_dummy_parquet(dummy_file)
    
    loader = DataLoader(str(dummy_file))
    
    # 2. Execution
    df = loader.load()
    
    # 3. Assertions
    # Check dimensions
    assert df.height == 100
    assert df.width == 16
    
    # Check Type Optimization
    # Features should be Float32 (down from Float64)
    assert df.schema['f0'] == pl.Float32
    assert df.schema['f11'] == pl.Float32
    
    # Flags should be Int8 (down from Int64)
    assert df.schema['treatment'] == pl.Int8
    assert df.schema['conversion'] == pl.Int8
    
    print("\nTest Passed: Data loaded and types optimized successfully.")

if __name__ == "__main__":
    # Simple runner without installing pytest CLI
    from pathlib import Path
    import tempfile
    import shutil
    
    # Create a temp dir manually for the script runner
    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_data_loader_optimization(temp_dir)
    finally:
        shutil.rmtree(temp_dir)