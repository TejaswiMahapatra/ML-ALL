import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model

# load the dataset
df = pd.read_csv("canada_per_capita_income.csv")
print(df.info())
new_df = df.drop('per capita income (US$)', axis='columns')
print(new_df)
pci = df[['per capita income (US$)']]
print(pci)
# create and fit the regression model
reg = linear_model.LinearRegression()
reg.fit(new_df, pci)
new_reg=reg.predict([[2020]])
print(new_reg)
