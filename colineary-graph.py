import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

# Assuming df is your DataFrame and 'column3' is the column you want to compare with others
column3_data = df['TotalMinutesAsleep']

# Get the column names starting from the 4th column
other_columns = df.columns[3:]

# Create a new DataFrame with 'column3' and the other columns
new_df = df[['TotalMinutesAsleep'] + list(other_columns)]

# Calculate the correlation matrix
correlation_matrix = new_df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()
