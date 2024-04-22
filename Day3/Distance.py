import csv
from math import sqrt
import numpy as np
import pandas as pd

df = pd.read_csv('Dataset_Day3.csv')

print(df)

df['distance'] = df.apply(lambda x: sqrt((x['dropoff_latitude'] - x['pickup_latitude']) ** 2 + (
            x['dropoff_longitude'] - x['pickup_longitude']) ** 2), axis=1)

print(df)


