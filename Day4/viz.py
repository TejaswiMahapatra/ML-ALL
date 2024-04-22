import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""2.	Make an in-depth report on the variables using descriptive statistics and also portray the relationship 
between each variables using visualizations"""
df = pd.read_csv('Dataset_Day4.csv')

skewness = df.skew()
print(skewness)

missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
df["TakeHome"].fillna(df["TakeHome"].median(), inplace=True)
df["Final"].fillna(df["Final"].median(), inplace=True)
print(df.info())
descriptive_statistics = df.describe()
print(descriptive_statistics)
# box-plot
Viz_data = df
sns.boxplot(data=Viz_data["Final"], orient='v')
# sns.boxplot(data=Viz_data["Tutorial"], orient='v')
# sns.boxplot(data=Viz_data["Midterm"], orient='v')
plt.show()
# pair-plot(since we're doing it for the entire dataframe)
sns.pairplot(Viz_data)
plt.show()
# heatmap
sns.heatmap(Viz_data.corr())
plt.show()

