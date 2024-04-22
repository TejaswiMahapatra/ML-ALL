import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset_Day6.csv')
print(df.info())
temp = df[["Furnishing", "Status", "Transaction", "Type"]]
# this function converts all categorical variables in the dataframe to one hot encoded variables
new_encoded_data = pd.get_dummies(temp)
print(new_encoded_data)




