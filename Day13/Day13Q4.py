import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# Load the dataset
df = pd.read_csv('Dataset_Day13.csv')
df.info()
# pair plot for additional insight
sns.pairplot(df)
plt.show()
# calculate missing-value percentage
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
df.boxplot(column=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
plt.title('Box Plot')
plt.show()
OutlierData = pd.DataFrame()
temp = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    Q1 = temp[col].quantile(0.25)  # Gives 25th Percentile or Q1
    Q3 = temp[col].quantile(0.75)  # Gives 75th Percentile or Q3

    IQR = Q3 - Q1

    UpperBound = Q3 + 1.5 * IQR
    LowerBound = Q1 - 1.5 * IQR

    OutlierData[col] = temp[col][(temp[col] < LowerBound) | (temp[col] > UpperBound)]
    print(len(OutlierData))
# group the dataset based on the "Species" column
descriptive_stats = df.groupby('Species').describe()
print(descriptive_stats)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# specify the number of clusters
k = 3
# create a kmeans instance
km = KMeans(n_clusters=k, n_init=25, random_state=1234)
# fit the data to the kmeans model
km.fit(X)
# get the cluster labels for each data point
cluster_labels = km.labels_
# the total within cluster sum of squares
clusterWCSS = km.inertia_
print(cluster_labels)
print(clusterWCSS)
# tabulation of the size of the clusters
pd.Series(km.labels_).value_counts().sort_index()
# 'km.cluster_centers_' :gives the cluster centroids
cluster_centers = pd.DataFrame(km.cluster_centers_,
                               columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
print(cluster_centers)
# to get the correct centroids, we need to un-scale the data,
cluster_centers_unscaled = pd.DataFrame()
for i in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    cluster_centers_unscaled[i] = (cluster_centers[i] * df[i].std()) + df[i].mean()
print(cluster_centers_unscaled)
wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=25, random_state=1234)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(2, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
silhouette = []
for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(X)
    silhouette.append(silhouette_score(X, km.labels_))

plt.plot(range(2, 11), silhouette)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()
calinski = []
for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(X)
    calinski.append(calinski_harabasz_score(X, km.labels_))

plt.plot(range(2, 11), calinski)
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski Harabasz Score')
plt.title('Calinski Harabasz Score')
plt.show()


