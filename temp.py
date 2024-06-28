import pandas as pd
import numpy as np
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


# Step 3: Define a function to remove collinear features based on a given threshold
def remove_collinear_features(features, threshold):
    correlation_matrix = features.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return features.drop(columns=to_drop), to_drop


# Step 4: Test different thresholds to achieve the highest R^2 score
best_threshold = None
best_r2 = -np.inf
best_features = None

thresholds = np.arange(0.6, 0.95, 0.05)  # Adjust the range and step as needed

for threshold in thresholds:
    reduced_features, dropped_columns = remove_collinear_features(features, threshold)

    # Prepare the data
    X = reduced_features.values
    y = df[target].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'kernel': ['poly'],
        'C': [0.1, 1, 10, 100],
        'degree': [2, 3, 4],
        'epsilon': [0.001, 0.01, 0.1],
        'coef0': [0, 1, 10],
        'gamma': ['scale', 'auto']
    }

    # Perform Grid Search Cross-Validation
    svm = SVR()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=0, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Retrieve the best model and evaluate
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    print(f'Threshold: {threshold:.2f}, R^2 Score: {r2:.3f}, Dropped Columns: {dropped_columns}')

    if r2 > best_r2:
        best_r2 = r2
        best_threshold = threshold
        best_features = reduced_features

# Step 5: Use the best threshold to prepare the final data
X = best_features.values
y = df[target].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['poly'],
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'epsilon': [0.001, 0.01, 0.1],
    'coef0': [0, 1, 10],
    'gamma': ['scale', 'auto']
}

# Perform Grid Search Cross-Validation
svm = SVR()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Retrieve the best model and evaluate
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Best Threshold: {best_threshold:.2f}')
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse:.3f}')
print(f'R^2 Score: {r2:.3f}')
