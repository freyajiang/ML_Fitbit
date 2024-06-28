import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

cols = df.columns[2:]
sns.pairplot(df[cols])
plt.show()
