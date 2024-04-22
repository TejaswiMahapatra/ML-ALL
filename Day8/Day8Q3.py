import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math

df = pd.read_csv('Dataset_Day7.csv')
print(df.info())
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
skewness = df.skew()
print(skewness)
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

df_OutlierFree = df.drop(OutlierData.index, axis=0)
df_OutlierFree.info()

X = df_OutlierFree.drop('Outcome', axis=1)  # all columns except 'Outcome'
y = df_OutlierFree['Outcome']  # target Variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=203)
unique_classes = np.unique(y)
print('Number of unique classes:', len(unique_classes))
k_start = int(math.sqrt(len(X_train)))
print(k_start)
metric_start = 'euclidean'

knn = KNeighborsClassifier(n_neighbors=k_start, metric=metric_start)
# fit the model
knn = knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Model Performance metrics are as below:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=1))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))

k_values = np.arange(2,25)
metric_values = ['euclidean', 'manhattan', 'hamming']
prec = []
rec = []
acc = []
PerfData = pd.DataFrame(columns=['Nearest Neighbor', 'Distance Metric', 'Precision', 'Recall', 'Accuracy'])

for dm in metric_values:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=dm)
        knn = knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        row = [[k, dm, precision_score(y_test, y_pred, zero_division=1), recall_score(y_test, y_pred),
                accuracy_score(y_test, y_pred)]]

        df2 = pd.DataFrame(row, columns=['Nearest Neighbor', 'Distance Metric', 'Precision', 'Recall', 'Accuracy'])
        PerfData = pd.concat([PerfData, df2], ignore_index=True)

print(PerfData.tail())
precision = PerfData['Precision']
recall = PerfData['Recall']
PerfData["F1 Score"] = (2 * PerfData["Precision"] * PerfData["Recall"]) / (PerfData["Precision"] + PerfData["Recall"])
print(PerfData[PerfData['F1 Score'] == max(PerfData['F1 Score'])])
# Plot Precision & Recall vs k Curve
plt.plot(k_values, PerfData[PerfData["Distance Metric"]=='euclidean']['Precision'], label='Precision')
plt.plot(k_values, PerfData[PerfData["Distance Metric"]=='euclidean']['Recall'], label='Recall')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Precision / Recall')
plt.title('Precision & Recall vs k Curve')
plt.legend(loc='upper right')
plt.show()
