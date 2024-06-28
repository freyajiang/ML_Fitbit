import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Step 1: Load the data
df = pd.read_csv('merged.csv')

# Step 2: Drop unnecessary columns and prepare target and features
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])

# Step 3: Identify and iteratively remove collinear features
threshold = 0.7
# threshold = 0.95
# threshold = 0.90
# threshold = 0.85
# threshold = 0.80
# threshold = 0.75
# threshold = 0.70
# threshold = 0.65
# threshold = 0.60
iteration = 1

while True:
    correlation_matrix = features.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    if not to_drop:
        break

    print(f"Iteration {iteration} - Columns to be dropped due to high collinearity:")
    print(to_drop)
    features = features.drop(columns=to_drop)
    iteration += 1

# Step 4: Prepare the data
X = features.values
y = df[target].values

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['poly'],
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'epsilon': [0.001, 0.01, 0.1],
    'coef0': [0, 1, 10],
    'gamma': ['scale', 'auto']
}

# Step 8: Perform Grid Search Cross-Validation
svm = SVR()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Retrieve the best model and evaluate
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse:.3f}')
print(f'R^2 Score: {r2:.3f}')

# Step 9: Calculate and print feature importance
result = permutation_importance(best_svm, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance = result.importances_mean
std = result.importances_std

feature_names = features.columns

print("\nFeature importance:")
for i in np.argsort(importance)[::-1]:
    print(f"{feature_names[i]:<30} {importance[i]:.4f} +/- {std[i]:.4f}")
