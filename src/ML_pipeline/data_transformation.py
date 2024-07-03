import pandas as pd 
import numpy as np 
from sklearn import preprocessing 

def cat_to_num(df, col, method='default', values=None):
    if method == 'default':
        label_encoder = preprocessing.LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        return df
    elif method == 'custom':
        if values is not None:
            df[col] = df[col].map(values).fillna(df[col])
        return df
    else:
        raise ValueError("Only these options for method are allowed : ['default','custom']")
    