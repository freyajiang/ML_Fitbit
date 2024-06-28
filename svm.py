import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('merged.csv')
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])
X = features.values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Linear Model: Mean Squared Error: {mse:.2f}')
print(f'Linear Model: R^2 Score: {r2:.2f}')

# RBF Model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RBF Model: Mean Squared Error: {mse:.2f}')
print(f'RBF Model: R^2 Score: {r2:.2f}')

# Polynomial Model
svm_model = SVR(kernel='poly')
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Polynomial Model: Mean Squared Error: {mse:.2f}')
print(f'Polynomial Model: R^2 Score: {r2:.2f}')