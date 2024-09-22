import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def percentage_missing_values(df):
    total_number_cells = df.shape[0]
    countMissing = df.isnull().sum()
    # totalMissing = countMissing.sum()
    return f"The telecom contains {round(((countMissing/total_number_cells) * 100), 2)}% missing values."


# def fill_null_values(df):
#     for column in df.columns:
#         if df[column].dtype == 'object' and df[column].dtype == 'category':
#             # Fill missing values with the previous value (forward fill)
#             df[column].fillna(method='ffill', inplace=True)
#         elif df[column].dtype == 'float64' and df[column].dtype == 'int64':
#             # Fill missing values with 0
#             df[column].fillna(0, inplace=True)
#     return df

def fill_null_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            # Fill missing values with the previous value (forward fill)
            df[column].fillna(method='ffill', inplace=True)
        elif df[column].dtype == 'float64' or df[column].dtype == 'int64':
            # Fill missing values with 0
            df[column].fillna(0, inplace=True)
    return df.info()
