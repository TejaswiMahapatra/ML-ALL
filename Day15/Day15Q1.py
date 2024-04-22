import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dataset_Day15.csv')
df.info()
# need to preprocess the data and convert it into a list of lists (transactions) before proceeding with the steps
# Convert each row into a single transaction (a list of items)
transactions = df.values.tolist()
# Display the first few transactions to check if they are correctly converted
print(transactions[:2])
# Remove all null or empty values from the list of lists
transactions = [[item for item in transaction if pd.notna(item)] for transaction in transactions]
print(transactions[:2])

