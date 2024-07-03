import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import math
import joblib 

def train_model(x_train, x_test, y_train, y_test, model_name, path):
    model_dict = {
        'LinearReg': LinearRegression,
        'SGDRegr' :SGDRegressor,
        'RFReg' : RandomForestRegressor,
        'DTReg' : DecisionTreeRegressor,
    }
    if model_name not in list(model_dict.keys()):
        raise ValueError(
            f"Only these options for model are allowed: {list(model_dict.keys())}"
        )
    model = model_dict[model_name]()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    joblib.dump(model, path)
    print("Model has been saved as pickle at"+str(path))
    return pred