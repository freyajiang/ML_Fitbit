import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

df = pd.read_csv('merged.csv')
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

all_predictors = model.pvalues.index[1:]  # Exclude the constant
predictors_included = []

adjusted_r2_values = {}

for predictor in all_predictors:
    predictors_included.append(predictor)
    model = sm.OLS(y_train, X_train[predictors_included + ['const']]).fit()
    adjusted_r2_values[predictor] = model.rsquared_adj


top_three_predictors = sorted(adjusted_r2_values, key=adjusted_r2_values.get, reverse=True)[:3]
print("\nTop five predictors based on adjusted R-squared values:")
for predictor in top_three_predictors:
    print(f"Predictor: {predictor}, Adjusted R-squared: {adjusted_r2_values[predictor]}")

print("\nTrained model's formula:")
print(f"{target} = {' + '.join([f'{coef} * {feat}' for coef, feat in zip(model.params, model.params.index)])}")

#### Analyze for Colinearity ####
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

features['Intercept'] = 1
vif = pd.DataFrame()
vif["Feature"] = features.columns
vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

print("\n\n")
print(vif)
