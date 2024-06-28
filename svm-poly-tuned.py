import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
df = pd.read_csv('merged.csv')

# Step 2: Drop unnecessary columns and prepare target and features
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])
X = features.values
y = df[target].values

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['poly'],
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'epsilon': [0.001, 0.01, 0.1],
    'coef0': [0, 1, 10],
    'gamma': ['scale', 'auto']
}

# Step 6: Perform Grid Search Cross-Validation
svm = SVR()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
# use verbose 3 to see more progress information.
grid_search.fit(X_train_scaled, y_train)

# Step 7: Retrieve the best model and evaluate
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse:.3f}')
print(f'R^2 Score: {r2:.3f}')
