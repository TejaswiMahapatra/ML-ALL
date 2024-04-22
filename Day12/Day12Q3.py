import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
# a. Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred = nb_clf.predict(X_test)

print("Naive Bayes Classifier - Default Model Performance Metrics:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))

# b. k-Fold Cross Validation
nb_clf_cv = GaussianNB()
cv_scores = cross_val_score(nb_clf_cv, X, y, cv=5, scoring='f1')

print("Naive Bayes Classifier - Cross-Validated F1-Score:", np.mean(cv_scores))

# c. Adaboost with Naive Bayes
adaboost_nb_clf = AdaBoostClassifier(n_estimators=50, estimator=nb_clf, learning_rate=1.0)
adaboost_nb_clf.fit(X_train, y_train)
y_pred_adaboost = adaboost_nb_clf.predict(X_test)

print("Adaboost with Naive Bayes :")
print("Accuracy: ", accuracy_score(y_test, y_pred_adaboost))
print("Precision: ", precision_score(y_test, y_pred_adaboost))
print("Recall: ", recall_score(y_test, y_pred_adaboost))
print("F1-Score: ", f1_score(y_test, y_pred_adaboost))

