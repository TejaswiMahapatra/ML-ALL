import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('Dataset_Day13.csv')
df.info()
# pair plot for additional insight
sns.pairplot(df)
plt.show()
# calculate missing-value percentage
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
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
# Scale the data
# df[columns] = MinMaxScaler().fit_transform(df[columns])
# print(df.info())
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# Fit the data to the DBSCAN model
cluster_labels = DBSCAN().fit_predict(X)
# Add the cluster labels to the original DataFrame
df['DBSCAN_Cluster'] = cluster_labels
# Show the species distribution in each of the default clusters
species_distribution = df.groupby(['DBSCAN_Cluster', 'Species']).size().reset_index(name='Count')
# Plot the distribution
sns.barplot(x='DBSCAN_Cluster', y='Count', hue='Species', data=species_distribution)
plt.xlabel('DBSCAN Cluster')
plt.ylabel('Count')
plt.title('Species Distribution in DBSCAN Clusters')
plt.show()
k = 7
nn = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nn.kneighbors(X)
distances = np.sort(distances, axis=0)[:, 1]
plt.plot(distances)
plt.axhline(y=0.6, color='r', ls="--")
plt.xlabel('Index')
plt.ylabel('k-Distance')
plt.title('K-Distance Graph')
plt.show()
optimal_eps = float(input("Enter the optimal eps value based on the plot: "))
dbscan = DBSCAN(eps=optimal_eps, min_samples=3)
cluster_labels = dbscan.fit_predict(X)
df['DBSCAN_Cluster'] = cluster_labels
species_distribution = df.groupby(['DBSCAN_Cluster', 'Species']).size().reset_index(name='Count')
sns.barplot(x='DBSCAN_Cluster', y='Count', hue='Species', data=species_distribution)
plt.xlabel('DBSCAN Cluster')
plt.ylabel('Count')
plt.title('Species Distribution in DBSCAN Clusters')
plt.show()
print(f"Optimal eps value: {optimal_eps:.3f}")
min_samples_values = range(2, 11)
silhouette_scores = []
for min_samples in min_samples_values:
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    # Check for single cluster case, as silhouette_score requires at least 2 clusters
    if len(set(cluster_labels)) > 1:
        silhouette_scores.append(silhouette_score(X, cluster_labels))
    else:
        silhouette_scores.append(-1)
plt.plot(min_samples_values, silhouette_scores)
plt.xlabel('min_samples')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different min_samples')
plt.grid()
plt.show()
# Find the most optimal 'min_samples' value that gives the highest silhouette score
optimal_min_samples = min_samples_values[np.argmax(silhouette_scores)]
print(f"Optimal min_samples value: {optimal_min_samples}")
dbscan = DBSCAN(eps=optimal_eps, min_samples=3)
cluster_labels = dbscan.fit_predict(X)
df['DBSCAN_Cluster'] = cluster_labels
species_distribution = df.groupby(['DBSCAN_Cluster', 'Species']).size().reset_index(name='Count')
sns.barplot(x='DBSCAN_Cluster', y='Count', hue='Species', data=species_distribution)
plt.xlabel('DBSCAN Cluster')
plt.ylabel('Count')
plt.title('Species Distribution in DBSCAN Clusters')
plt.show()

# Identify outliers with cluster label '-1'
outliers = X[cluster_labels == -1]
print("Outliers:")
print(outliers)