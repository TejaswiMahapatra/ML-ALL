import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('Dataset_Day6.csv')
print(df.info())
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
df["Bathroom"].fillna(df["Bathroom"].median(), inplace=True)
df["Furnishing"].fillna(df["Furnishing"].mode()[0], inplace=True)  # mode() because this is categorical data
df["Parking"].fillna(df["Parking"].median(), inplace=True)
df["Type"].fillna(df["Type"].mode()[0], inplace=True)
print(df.info())
X = df.drop('Price', axis=1)  # all columns except 'Price'
y = df['Price']  # target Variable
temp = df[["Furnishing", "Status", "Transaction", "Type"]]
# this function converts all categorical variables in the dataframe to one hot encoded variables
new_encoded_data = pd.get_dummies(temp)
# Concatenate encoded categorical variables with remaining features
X = pd.concat([X.select_dtypes(exclude=['object']), new_encoded_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
lm = LinearRegression()
lm = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
print(lm.coef_)  # scale parameter
print(lm.intercept_)  # intercept parameter
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mae = mean_absolute_error(y_test, y_pred)
print("R2 Score:", r2)
print("Adjusted R2 Score:", adjusted_r2)
print("Mean Absolute Error (MAE):", mae)
