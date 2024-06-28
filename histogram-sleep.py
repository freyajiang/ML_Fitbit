import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

plt.hist(df['TotalMinutesAsleep'], bins=10)
plt.title('Histogram of TotalMinutesAsleep')
plt.show()
