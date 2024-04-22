import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
temp = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]]
for col in ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]:
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)
unique_classes = np.unique(y)
print('Number of unique classes:', len(unique_classes))
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

#  Create a svm Classifier
svm_clf = svm.SVC(kernel=kernel_values[0], C=1, gamma=0.1)  # Linear Kernel

# Train the model using the training sets
svm_clf = svm_clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = svm_clf.predict(X_test)

print("Model Performance metrics are as below :-\n")
print("Accuracy is " + str(accuracy_score(y_test, y_pred)))
print("Precision is " + str(precision_score(y_test, y_pred, zero_division=1)))
print("Recall is " + str(recall_score(y_test, y_pred)))
print("F1-Score is " + str(f1_score(y_test, y_pred)))

# C_values = [0.001,0.01,0.1,1,10,100,1000] # C of 1 is a good starting point || C > 0
C_values = [0.01, 0.1, 1, 10]
# gamma_values = [0.1,0.5,0.9] || Gamma > 0 || starting value : gamma = 1/len(X) = 1/no. of rows in the dataset
gamma_values = [0.1, 1 / len(X)]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

PerfData = pd.DataFrame(columns=['Kernel Type', 'C', 'Gamma', 'Precision', 'Recall', 'Accuracy', 'F1-Score'])

for k in kernel_values:
    for c in C_values:
        for g in gamma_values:
            svm_clf = svm.SVC(kernel=k, C=c, gamma=g)
            svm_clf = svm_clf.fit(X_train, y_train)
            y_pred = svm_clf.predict(X_test)

            row = [
                [k, c, g, precision_score(y_test, y_pred, zero_division=1), recall_score(y_test, y_pred), accuracy_score(y_test, y_pred),
                 f1_score(y_test, y_pred)]]
            df2 = pd.DataFrame(row, columns=['Kernel Type', 'C', 'Gamma', 'Precision', 'Recall', 'Accuracy', 'F1-Score'])
            PerfData = pd.concat([PerfData, df2], ignore_index=True)

print(PerfData.head(10))
print(PerfData[PerfData['F1-Score'] == max(PerfData['F1-Score'])])
plt.plot(kernel_values, PerfData[PerfData['Kernel Type'] == 'linear']['Precision'], label='Precision')
plt.plot(kernel_values, PerfData[PerfData['Kernel Type'] == 'linear']['Recall'], label='Recall')
plt.scatter(kernel_values, PerfData[PerfData['Kernel Type'] == 'linear']['F1-Score'], label='F1-Score')
plt.xlabel('Kernel type')
plt.ylabel('Score')
plt.title('Precision & Recall & F-1 Score vs kernel type')
plt.legend(loc='upper right')
plt.show()
svm_best_kernel = SVC(kernel=best_kernel)
precisions = []
recalls = []
f1_scores = []
C_values = np.arange(0.1, 10, 0.1)
for C in C_values:
    svm_best_kernel.set_params(C=C)
    svm_best_kernel.fit(X_train, y_train)
    y_pred = svm_best_kernel.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

best_C = C_values[np.argmax(f1_scores)]
print(f"The C value with the highest F1-score for kernel type '{best_kernel}' is: {best_C}")

plt.plot(C_values, precisions, label='Precision')
plt.plot(C_values, recalls, label='Recall')
plt.plot(C_values, f1_scores, label='F1-Score')
plt.xlabel('C')
plt.ylabel('Score')
plt.title(f"Precision, Recall, and F1-Score vs C for Kernel Type '{best_kernel}'")
plt.legend()
plt.show()


