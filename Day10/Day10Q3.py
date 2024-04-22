import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Dataset_Day7.csv')

# Handle missing values
df["Glucose"].fillna(df["Glucose"].median(), inplace=True)
df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
df["BMI"].fillna(df["BMI"].median(), inplace=True)
df["Outcome"].fillna(df["Outcome"].mean(), inplace=True)

# Split the data into training and testing sets
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Part (a) - Default model performance metrics
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

print("Model Performance metrics (Default Parameters):\n")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))

# Part (b) - Plot Precision & Recall vs max_leaf_nodes
max_leaf_nodes = np.arange(2, 21)
precisions = []
recalls = []
f1_scores = []

for nodes in max_leaf_nodes:
    dt_clf = DecisionTreeClassifier(max_leaf_nodes=nodes)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

best_max_leaf_node = max_leaf_nodes[np.argmax(f1_scores)]
print("Best max leaf nodes for highest F1-Score:", best_max_leaf_node)
plt.plot(max_leaf_nodes, precisions, label='Precision')
plt.plot(max_leaf_nodes, recalls, label='Recall')
plt.plot(max_leaf_nodes, f1_scores, label='f1_Score')
plt.xlabel('max_leaf_nodes')
plt.ylabel('Score')
plt.title('Precision & Recall vs max_leaf_nodes')
plt.legend()
plt.show()

# Part (c) - Plot Precision & Recall vs max_depth
max_depths = np.arange(2, 21)
precisions = []
recalls = []
f1_scores = []

for depth in max_depths:
    dt_clf = DecisionTreeClassifier(max_depth=depth)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

best_max_depth = max_depths[np.argmax(f1_scores)]
print("Best max depth for highest F1-Score:", best_max_depth)
plt.plot(max_depths, precisions, label='Precision')
plt.plot(max_depths, recalls, label='Recall')
plt.plot(max_depths, f1_scores, label='f1_Score')
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.title('Precision & Recall vs max_depth')
plt.legend()
plt.show()
