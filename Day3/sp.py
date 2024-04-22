import csv
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset_Day3.csv')
df['distance'] = df.apply(lambda x: sqrt((x['dropoff_latitude'] - x['pickup_latitude']) ** 2 + (
            x['dropoff_longitude'] - x['pickup_longitude']) ** 2), axis=1)
print(df)

plt.scatter(df["distance"], df["fare_amount"])
plt.title('Simple Scatter-plot between distance & fare_amount')
plt.xlabel('X-Distance')
plt.ylabel('Y-fare_amount')

plt.show()
