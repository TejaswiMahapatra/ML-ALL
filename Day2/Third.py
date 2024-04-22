import pandas as pd
import numpy as np
import csv

df = pd.read_csv('Dataset_Day2.csv')
df_HighSodLowProt = df[(df['sodium'] > 100) & (df['protein'] < 3)]
# average calories by mfr
average = df_HighSodLowProt.groupby(['mfr'])['calories'].mean()
# Print
print("Average Calories by mfr:")
print(average)
# mfr with the highest average calories
highest_average = average.idxmax()
# Print
print("mfr with Highest Average Calories:", highest_average)
