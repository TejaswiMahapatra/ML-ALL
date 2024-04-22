import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge


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
alpha = [1e-50, 1e-20, 1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]
print(alpha)

for param in alpha:
    ridgeModel = Ridge(alpha=param)
    ridgeModel.fit(X_train, y_train)

    y_pred = ridgeModel.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)
    print(r2)
    print("alpha = {}".format(param))
    print("mse for above alpha = {}".format(mse))
    best_r2 = 0
    best_alpha = 0
    best_mse = mse

    if r2 > best_r2:
        best_r2 = r2

    if mse < best_mse:
        best_mse = mse
        best_alpha = param

print("Best value of alpha/lambda = {}".format(best_alpha))
print("MSE for this alpha/lambda = {}".format(best_mse))
print("Best R square score = {}".format(best_r2))

