import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

column3_data = df['TotalMinutesAsleep']

# Get the column names starting from the 4th column
other_columns = df.columns[3:]

for column in other_columns:
    other_column_data = df[column]

    # Calculate correlation
    correlation = column3_data.corr(other_column_data)

    print(f'Correlation between TotalMinutesAsleep and {column}: {correlation}')
