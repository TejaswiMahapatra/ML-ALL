import pandas as pd
import numpy as np
import csv

# Read the dataset(pandas DataFrame)
df = pd.read_csv('Dataset_Day2.csv')
# Create a new DataFrame(satisfying the condition)
df_HighSodLowProt = df[(df['sodium'] > 100) & (df['protein'] < 3)]
# Print the new DataFrame
print(df_HighSodLowProt)
