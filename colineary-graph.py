import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

column3_data = df['TotalMinutesAsleep']
other_columns = df.columns[3:]
new_df = df[['TotalMinutesAsleep'] + list(other_columns)]

correlation_matrix = new_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()
