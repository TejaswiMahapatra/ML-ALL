import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
1.	Find and treat all the missing values. Rows or columns with missing values must not be dropped.
"""
df = pd.read_csv('Dataset_Day4.csv')
print(df.info())
skewness = df.skew()
print(skewness)
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
df["TakeHome"].fillna(df["TakeHome"].median(), inplace=True)
print(df.info())
df["Final"].fillna(df["Final"].median(), inplace=True)
print(df.info())

