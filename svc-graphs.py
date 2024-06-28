# svc-graphs.py
# Graphs the data points and look for SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('merged.csv')

# Drop unnecessary columns and prepare target and features
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])
X = features.values
y = df[target].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.01, 0.001],
    'coef0': [0, 1, 10],
    'degree': [2, 3, 4]  # Only relevant for polynomial kernel
}

# Perform Grid Search Cross-Validation with Parallelization and Verbose Output
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

# Function to plot decision boundary for linear and nonlinear SVC
def plot_decision_boundary(model, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)  # Background colors
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)  # Contour lines
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.show()

# For visualization, we need 2D data, we can use PCA to reduce the dimensionality for plotting
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train linear SVC for visualization
linear_svc = SVC(kernel='linear')
linear_svc.fit(X_train_pca, y_train)

# Plot linear SVC decision boundary
plot_decision_boundary(linear_svc, X_train_pca, y_train, 'Linear SVC Decision Boundary')

# Train nonlinear SVC (RBF kernel) for visualization
rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(X_train_pca, y_train)

# Plot nonlinear SVC decision boundary
plot_decision_boundary(rbf_svc, X_train_pca, y_train, 'Nonlinear SVC (RBF) Decision Boundary')

# Train polynomial SVC for visualization
poly_svc = SVC(kernel='poly', degree=3)  # You can change the degree as needed
poly_svc.fit(X_train_pca, y_train)

# Plot polynomial SVC decision boundary
plot_decision_boundary(poly_svc, X_train_pca, y_train, 'Polynomial SVC Decision Boundary (Degree 3)')

# Nonlinear SVC with Cross-Validation Scores
# Perform Grid Search Cross-Validation for Nonlinear SVC (e.g., RBF kernel)
param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10, 100]
}

rbf_svc = SVC(kernel='rbf')
grid_search_svc = GridSearchCV(estimator=rbf_svc, param_grid=param_grid_svc,
                               scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_search_svc.fit(X_train_pca, y_train)

# Plot cross-validation scores
def plot_cv_scores(grid_search, param_name, title):
    scores = grid_search.cv_results_['mean_test_score']
    params = grid_search.cv_results_['param_' + param_name]
    plt.figure(figsize=(10, 6))
    plt.plot(params, scores, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean Cross-Validation Score')
    plt.title(title)
    plt.show()

# # Plot cross-validation scores for C and gamma
# plot_cv_scores(grid_search_svc, 'C', 'Cross-Validation Scores for C (RBF Kernel)')
# plot_cv_scores(grid_search_svc, 'gamma', 'Cross-Validation Scores for Gamma (RBF Kernel)')

# Display cross-validation results for the nonlinear SVC in table format
cv_results_svc_df = pd.DataFrame(grid_search_svc.cv_results_)
print(cv_results_svc_df[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score', 'rank_test_score']])
