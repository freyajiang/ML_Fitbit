import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

columns = df.columns[2:]

for column in columns:
    plt.figure(figsize=(12, 6))
    plt.hist(df[column].dropna(), bins=10)
    plt.title(f'Histogram of {column}')
    plt.show()
