import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

df = pd.read_csv('Dataset_Day7.csv')
print(df.info())
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
skewness = df.skew()
print(skewness)
df["Pregnancies"].fillna(df["Pregnancies"].median(), inplace=True)
df["Glucose"].fillna(df["Glucose"].median(), inplace=True)
df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
df["BMI"].fillna(df["BMI"].median(), inplace=True)
df["Outcome"].fillna(df["Outcome"].mean(), inplace=True)
print(df.info())
OutlierData = pd.DataFrame()
temp = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]]
for col in ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]:
    Q1 = temp[col].quantile(0.25)  # Gives 25th Percentile or Q1
    Q3 = temp[col].quantile(0.75)  # Gives 75th Percentile or Q3

    IQR = Q3 - Q1

    UpperBound = Q3 + 1.5 * IQR
    LowerBound = Q1 - 1.5 * IQR

    OutlierData[col] = temp[col][(temp[col] < LowerBound) | (temp[col] > UpperBound)]
    print(len(OutlierData))
    df_OutlierFree = df.drop(OutlierData.index, axis=0)
    df_OutlierFree.info()
# scaling on the selected continuous columns by subtracting the minimum value and dividing by the range (maximum -
# minimum) for each column.
continuous_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df_continuous = df_OutlierFree[continuous_columns]
df_scaled = df_OutlierFree.copy()
df_scaled[continuous_columns] = (df_continuous - df_continuous.min()) / (df_continuous.max() - df_continuous.min())
X = df_OutlierFree.drop('Outcome', axis=1)  # all columns except 'Outcome'
y = df_OutlierFree['Outcome']  # target Variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=203)
unique_classes = np.unique(y)
print('Number of unique classes:', len(unique_classes))
logreg = LogisticRegression(random_state=203)
logreg = logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)  # default threshold value is 0.5
y_scores = logreg.predict_proba(X_test)[:, 1]
# display(y_scores)

prec, rec, tre = precision_recall_curve(y_test, y_scores)

plt.plot(tre, prec[:-1], 'r--', label='Precision')
plt.plot(tre, rec[:-1], 'b--', label='Recall')

f_score = (2 * prec * rec) / (prec + rec)

plt.plot(tre, f_score[:-1], 'g--', label='F1-Score')

plt.xlabel('Threshold range')
plt.legend(loc='upper left')
plt.show()

index = np.where(f_score == max(f_score))
print("Optimum Threshold for max precision and recall is {}".format(tre[index]))
