import pandas as pd
import logging
from typing import Optional, List, Union

def read_parquet_optimized(
    path: str, 
    columns: Optional[List[str]] = None, 
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """
    Read parquet file with optimization for memory usage
    
    Args:
        path: Path to parquet file
        columns: List of columns to read (None for all)
        chunksize: If specified, read in chunks and return first chunk (for memory optimization)
    
    Returns:
        DataFrame
    """
    try:
        if chunksize:
            # Read in chunks for memory optimization
            df_chunks = pd.read_parquet(path, columns=columns, chunksize=chunksize)
            return pd.concat(df_chunks, ignore_index=True)
        else:
            return pd.read_parquet(path, columns=columns)
    except Exception as e:
        logging.error(f"Error reading parquet file {path}: {e}")
        raise