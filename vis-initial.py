# checking for problems, like blanks, in a visual way
import pandas as pd

df = pd.read_csv('initial.csv')

numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()


import matplotlib.pyplot as plt

missing_values = df.isnull().sum()
missing_values.plot(kind='bar')
plt.title('Number of Missing Values in Each Column')
plt.show()

import seaborn as sns

correlation_matrix = numeric_df.corr()
# correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Heatmap of Variable Correlations')
plt.show()

