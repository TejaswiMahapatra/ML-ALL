import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy.stats import boxcox

# Load the dataset
df = pd.read_csv("qsar_fish_toxicity.csv")

# Separate the target variable from the features
X = df.drop(columns=['LC50 [-LOG(mol/L)]'])
y = df['LC50 [-LOG(mol/L)]']

# Print the information about the dataset
print(X.info())

# Plot a pair plot to see the relationships between the different variables
sns.pairplot(X)
plt.show()

# Plot a heatmap to show the missing values in the dataset
plt.figure(figsize=(8, 6))
sns.heatmap(X.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# Calculate the skewness of each column
skewness = X.skew()
print(skewness)

# Check the distribution of the data in each column
for column in X.columns:
    print(column, X[column].describe())

# Calculate the missing-value percentage
missing_value_percent = X.isna().sum() / len(X) * 100
print("Missing Value Percentage:")
print(missing_value_percent)

# Impute the missing values using IterativeImputer
imputed_data = IterativeImputer(verbose=False).fit_transform(X)

# Convert the imputed data to a Pandas DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=X.columns)

# Perform winsorization on each column to handle outliers
winsorized_df = imputed_df.apply(lambda x: winsorize(x, limits=[0.1, 0.1]))

