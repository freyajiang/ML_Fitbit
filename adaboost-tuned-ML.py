import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(df.columns[[0, 1, 3]], axis=1)

target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1, 10],
}

ada_model = AdaBoostRegressor(random_state=42)

grid_search = GridSearchCV(estimator=ada_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_ada_model = grid_search.best_estimator_
print(f"Best parameters found: {best_params}")

y_pred = best_ada_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error after tuning: {mse}")
print(f"R-squared after tuning: {r2}")
print("\nBest AdaBoost Model:")
print(best_ada_model)