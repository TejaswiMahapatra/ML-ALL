import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

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
print(df_onehot.head())
# Find frequent item sets with minimum support of 0.02
frequent_itemsets = apriori(df_onehot, min_support=0.02, use_colnames=True)
print(frequent_itemsets)
# sort the values
frequent_itemsets.sort_values('support', ascending=False)
# Generate association rules with minimum threshold of 15%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.15)
# top 5 Sort rules based on conviction, leverage, and lift
top_rules_conviction = rules.sort_values(by='conviction', ascending=False).head(5)
top_rules_leverage = rules.sort_values(by='leverage', ascending=False).head(5)
top_rules_lift = rules.sort_values(by='lift', ascending=False).head(5)
# Find top 2 and bottom 2 rules based on Zhang's metric
top_rules_zhangs = rules.sort_values(by='zhangs_metric', ascending=False).head(2)
bottom_rules_zhangs = rules.sort_values(by='zhangs_metric').head(2)
print("Top 5 Antecedent -> Consequent Rules based on Conviction:")
print(top_rules_conviction)
print("Top 5 Antecedent -> Consequent Rules based on Leverage:")
print(top_rules_leverage)
print("Top 5 Antecedent -> Consequent Rules based on Lift:")
print(top_rules_lift)
print("Top 2 Antecedent -> Consequent Rules based on Zhang's Metric:")
print(top_rules_zhangs)
print("Bottom 2 Antecedent -> Consequent Rules based on Zhang's Metric:")
print(bottom_rules_zhangs)
# Filter the association rules DataFrame based on 'ground meat' and 'spaghetti'
filtered_rules = rules[(rules['antecedents'] == {'ground meat'}) & (rules['consequents'] == {'spaghetti'})]
# Print the lift value
print("Lift Value for the rule (ground meat -> spaghetti):")
print(filtered_rules['lift'].values[0])
