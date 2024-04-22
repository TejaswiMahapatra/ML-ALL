import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

"""
2.	Create a simple linear regression model that quantitatively relates ‘MEDV’ with ‘RM’. (10 marks)
a.	Share the model performance metrics and print the full regression model with coefficients.
b.	Use the model to predict the price of the house for ‘RM’ = 7
"""
df = pd.read_csv('Dataset_Day5.csv')
print(df.info())
X = df[['RM']]
y = df[['MEDV']]
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# calling and fitting the model on training data
lm = LinearRegression()
lm = lm.fit(X_train, y_train)
# predict using test set
y_pred = lm.predict(X_test)
print(lm.coef_)  # scale parameter
print(lm.intercept_)  # intercept parameter
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
rm_value = 7
price_prediction = lm.predict([[rm_value]])
print("Predicted price for 'RM' = 7:", price_prediction[0])
