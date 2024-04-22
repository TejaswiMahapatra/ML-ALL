import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Dataset_Day7.csv')

# Handle missing values
df["Glucose"].fillna(df["Glucose"].median(), inplace=True)
df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
df["BMI"].fillna(df["BMI"].median(), inplace=True)
df["Outcome"].fillna(df["Outcome"].mean(), inplace=True)

# Split the data into training and testing sets
X = df.drop('Outcome', axis=1)
y = df['Outcome']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Part (a) - Bagging on Decision Trees
# Note: Parameter - n_estimators stands for how many tree we want to grow
n_estimators = np.arange(2, 26)
f1_scores = []
accuracy = []

for n in n_estimators:
    bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n)
    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

print("Bagging (Decision Trees):")
print("Accuracy: ", accuracy)
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_scores)
best_n_estimators_bagging = n_estimators[np.argmax(f1_scores)]
print("Best n_estimators for Bagging: ", best_n_estimators_bagging)

plt.plot(n_estimators, f1_scores, label='F1-Score')
plt.plot(n_estimators, accuracy, label='Accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Score')
plt.title('F1-Score & Accuracy vs n_estimators (Bagging)')
plt.legend()
plt.show()

# Part (b) - Random Forest
f1_times_accuracy = []
for n in n_estimators:
    rf_clf = RandomForestClassifier(n_estimators=n)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    f1_times_accuracy.append(f1_score(y_test, y_pred) * accuracy_score(y_test, y_pred))

print("Random Forest:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))
best_n_estimators_random_forest = n_estimators[np.argmax(f1_scores)]
print("Best n_estimators for Random Forest: ", best_n_estimators_random_forest)
plt.plot(n_estimators, f1_times_accuracy)
plt.xlabel('n_estimators')
plt.ylabel('F1-Score * Accuracy')
plt.title('F1-Score * Accuracy vs n_estimators (Random Forest)')
plt.show()

# Part (c) - Adaboost on Decision Trees
adaboost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier())
adaboost_clf.fit(X_train, y_train)
y_pred = adaboost_clf.predict(X_test)

print("Adaboost (Decision Trees):")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))
best_n_estimators_adaboost = adaboost_clf.n_estimators
print("Best n_estimators for AdaBoost: ", best_n_estimators_adaboost)
