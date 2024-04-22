import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset_Day6.csv')
print(df.info())
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
df["Bathroom"].fillna(df["Bathroom"].median(), inplace=True)
df["Furnishing"].fillna(df["Furnishing"].mode()[0], inplace=True)  # mode() because this is categorical data
df["Parking"].fillna(df["Parking"].median(), inplace=True)
df["Type"].fillna(df["Type"].mode()[0], inplace=True)
print(df.info())
print(df[["Price"]])
OutlierData = pd.DataFrame()
temp = df[["Price"]]
for col in ["Price"]:
    Q1 = temp[col].quantile(0.25)  # Gives 25th Percentile or Q1
    Q3 = temp[col].quantile(0.75)  # Gives 75th Percentile or Q3

    IQR = Q3 - Q1

    UpperBound = Q3 + 1.5 * IQR
    LowerBound = Q1 - 1.5 * IQR

    OutlierData[col] = temp[col][(temp[col] < LowerBound) | (temp[col] > UpperBound)]
    df_OutlierFree = df.drop(OutlierData.index, axis=0)
print(OutlierData.index)
print(len(OutlierData))
print(df_OutlierFree[["Price"]])
print(df_OutlierFree[["Bathroom"]])
df_OutlierFree.info()
