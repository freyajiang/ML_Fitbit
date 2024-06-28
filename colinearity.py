import pandas as pd

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

column3_data = df['TotalMinutesAsleep']
other_columns = df.columns[3:]
new_df = df[['TotalMinutesAsleep'] + list(other_columns)]

correlation_matrix = new_df.corr()
print(correlation_matrix)
