import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
predictor_r2 = {}

predictors = df.columns.drop(target)
for predictor in predictors:
    model = smf.ols(f'{target} ~ {predictor}', data=X_train).fit()
    predictor_r2[predictor] = model.rsquared_adj

sorted_predictors = sorted(predictor_r2.items(), key=lambda item: item[1], reverse=True)
top_three = sorted_predictors[:3]

for predictor, r2 in top_three:
    print(f"Predictor: {predictor}, Adjusted R-squared: {r2:2f}")
