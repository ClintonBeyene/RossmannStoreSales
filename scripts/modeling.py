import numpy as np
import pandas as pd

def rmspe(y_true, y_pred, chunk_size=1000):
    # Ensure y_true and y_pred are numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    elif isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
        
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()
    elif isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()
    
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    
    # Avoid division by zero
    non_zero_indices = y_true!= 0
    
    # Calculate RMSPE for each chunk of data
    chunk_rmspes = []
    for i in range(0, len(y_true), chunk_size):
        chunk_y_true = y_true[i:i+chunk_size]
        chunk_y_pred = y_pred[i:i+chunk_size]
        chunk_non_zero_indices = non_zero_indices[i:i+chunk_size]
        
        chunk_errors = (chunk_y_true[chunk_non_zero_indices] - chunk_y_pred[chunk_non_zero_indices]) / chunk_y_true[chunk_non_zero_indices]
        chunk_rmspe = np.sqrt(np.mean(chunk_errors ** 2))
        chunk_rmspes.append(chunk_rmspe)
    
    # Calculate overall RMSPE
    return np.sqrt(np.mean(np.array(chunk_rmspes) ** 2)) 

