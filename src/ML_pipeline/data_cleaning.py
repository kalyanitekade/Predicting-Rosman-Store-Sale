import pandas as pd 
import numpy as np 

def cleaning(df, col, method, value =0):
    if method =="mean":
        mean_val = df[col].mean()
        df[col]= df[col].fillna(mean_val)
        return df 
    elif method =="median":
        median_val = df[col].median()
        df[col]=df[col].fillna(median_val)
        return df
    elif method =="mode":
        mode_val = df[col].mode()
        df[col]=df[col].fillna(mode_val)
        return df 
    elif method == 'value':
        df[col] = df[col].fillna(value)
        return df
    else:
        raise ValueError(
            f"Only these methods [mean, median, mode, 0] are available. Choose from them."
        )
        
        