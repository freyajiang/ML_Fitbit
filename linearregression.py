from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd


df = pd.read_csv('merged.csv')
df = df.drop(df.columns[[0, 1, 3]], axis=1)

target = 'TotalMinutesAsleep'
model = LinearRegression()
r2_values = {}

for column in df.columns:
    if column != target:
        model.fit(df[[column]], df[target])
        predictions = model.predict(df[[column]])
        r2_values[column] = r2_score(df[target], predictions)

for column, r2 in r2_values.items():
    print(f"R2 value for {column} vs {target}: {r2:3f}")

top_three_predictors = sorted(r2_values, key=r2_values.get, reverse=True)[:3]

print(f"\nTop three predictors for {target} based on R2 value:")
for predictor in top_three_predictors:
    print(f"{predictor}: {r2_values[predictor]:3f}")
