import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder

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
# converts the list of lists (transactions) into a one-hot encoded DataFrame, where each column represents an item, and each row represents a transaction
# Initialize TransactionEncoder
te = TransactionEncoder()
# Fit and transform the transactions to one-hot encoded DataFrame
onehot_encoded = te.fit_transform(transactions)
# Convert the one-hot encoded array to a DataFrame
df_onehot = pd.DataFrame(onehot_encoded, columns=te.columns_)
# Display the first few rows of the one-hot encoded DataFrame
print(df_onehot.head())

