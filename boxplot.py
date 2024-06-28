import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

# Assuming df is your DataFrame
columns = df.columns[3:]

for column in columns:
    plt.figure(figsize=(12, 6))
    plt.boxplot(df[column].dropna())
    plt.title(f'Box Plot of {column}')
    plt.show()
