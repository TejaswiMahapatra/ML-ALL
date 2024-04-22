import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
1.	Perform a complete data inspection including 
a.	Missing Data Treatment
b.	Descriptive Statistics of each variable (Eg. Boxplot, Histogram etc.)
c.	Visualization of all continuous variables

"""
df = pd.read_csv('Dataset_Day5.csv')
print(df.info())
skewness = df.skew()
print(skewness)
missing_value_percent = df.isna().sum() / len(df) * 100
print(missing_value_percent)
descriptive_statistics = df.describe()
print(descriptive_statistics)
# proportion of non-retail business acres per town (boxplot is also done for continuous variables)
sns.boxplot(data=df["INDUS"], orient='v')
plt.plot()
# sns.histplot(df, x='NOX')
# plt.plot()
# for continuous variables
sns.pairplot(df[['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT', 'MEDV']])
plt.show()
# plot between CRIM(per capita crime rate by town) and B(1000(bk-0.63)^2 bk is proportion of black people in town)
plt.scatter(df['B'], df['CRIM'])
plt.xlabel('B')
plt.ylabel('CRIM')
plt.title('Scatter Plot of B vs CRIM')
plt.show()
sns.heatmap(df.corr())
plt.show()
