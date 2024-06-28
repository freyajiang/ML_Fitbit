import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Heatmap of Variable Correlations')
plt.show()