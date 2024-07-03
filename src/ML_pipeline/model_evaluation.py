import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math 

def eval_model(y_test, y_pred):
    metrics={
        'r2_score':r2_score(y_test, y_pred),
        'MAE':mean_absolute_error(y_test, y_pred),
        'RMSE' : math.sqrt(mean_squared_error(y_test, y_pred))
    }
    return metrics