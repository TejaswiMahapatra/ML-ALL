import csv
from math import sqrt
import numpy as np
import pandas as pd

df = pd.read_csv('Dataset_Day3.csv')
df['distance'] = df.apply(lambda x: sqrt((x['dropoff_latitude'] - x['pickup_latitude']) ** 2 + (
        x['dropoff_longitude'] - x['pickup_longitude']) ** 2), axis=1)
print(df)
OutlierData = pd.DataFrame()
temp = df[["distance", "fare_amount", "passenger_count"]]
for col in ["distance", "fare_amount","passenger_count"]:
    Q1 = temp[col].quantile(0.25)  # Gives 25th Percentile or Q1
    Q3 = temp[col].quantile(0.75)  # Gives 75th Percentile or Q3

    IQR = Q3 - Q1

    UpperBound = Q3 + 1.5 * IQR
    LowerBound = Q1 - 1.5 * IQR

    OutlierData[col] = temp[col][(temp[col] < LowerBound) | (temp[col] > UpperBound)]
    print(len(OutlierData))
    df_OutlierFree = df.drop(OutlierData.index, axis=0)
    df_OutlierFree.info()

