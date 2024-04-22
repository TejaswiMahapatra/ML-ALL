import pandas as pd
import numpy as np
import csv

# load the dataset(pandas DataFrame)
df = pd.read_csv('Dataset_Day2.csv')
# Extract the columns
name_col = df['name']
mfr_col = df['mfr']
vitamins_col = df['vitamins']
# Unique values(using numpy.unique() function)
names = np.unique(name_col)
mfr = np.unique(mfr_col)
vitamins = np.unique(vitamins_col)
# Store in separate NumPy arrays
print(names)
print(mfr)
print(vitamins)
