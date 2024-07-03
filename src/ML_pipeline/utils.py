import pandas as pd 
import numpy as np 

def read_dataset(path):
    df = pd.read_csv(path)
    return df 

def merge_df(df1,df2, col_name):
    combined_data = pd.merge(df1,df2, on=col_name)
    return combined_data

def remove_outliers(df,col,thres):
    df=df.drop(df.loc[df[col]>thres].index)
    return df 

def year_from_date(df, date_col, new_col='year'):
    if new_col in df:
        raise KeyError(
            f"{new_col} already exists in the df")
    df[new_col]=pd.DatetimeIndex(df[date_col]).year
    return df

def month_from_date(df, date_col, new_col = 'month'):
    if new_col in df:
        raise KeyError(
            f"{new_col} already exists in the df"
        )
    df[new_col]=pd.DatetimeIndex(df[date_col]).month
    return df
    

    