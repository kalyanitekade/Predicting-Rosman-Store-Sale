import pandas as pd 
import numpy as np 


def feature_importance(features, model):
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        return dict(zip(features, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        # Linear models
        return dict(zip(features, model.coef_))
    else:
        return None