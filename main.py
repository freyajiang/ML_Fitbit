import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

import seaborn as sns

# Assuming df is your DataFrame
# correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Heatmap of Variable Correlations')
plt.show()

from scipy import stats

# Assuming df is your DataFrame and 'column3' is the column you want to compare with others
group1 = df['TotalMinutesAsleep']

# Get the column names starting from the 4th column
other_columns = df.columns[3:]

for column in other_columns:
    group2 = df[column]

    # Remove NaN values
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(group1_clean, group2_clean)

    print(f'T-statistic for TotalMinutesAsleep vs {column}: {t_statistic}, P-value: {p_value}')

# Assuming df is your DataFrame and 'column1' is the column you want to plot
plt.boxplot(df['TotalMinutesAsleep'])
plt.title('Box Plot of TotalMinutesAsleep')
plt.show()

# Assuming df is your DataFrame and 'column1' is the column you want to plot
plt.hist(df['TotalMinutesAsleep'], bins=10)
plt.title('Histogram of TotalMinutesAsleep')
plt.show()

