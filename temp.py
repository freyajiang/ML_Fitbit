import pandas as pd
from sklearn.model_selection import train_test_split
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

# Step 5: Train and evaluate SVM models with different kernels
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    svm_model = SVR(kernel=kernel)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{kernel.capitalize()} Model: Mean Squared Error: {mse:.2f}')
    print(f'{kernel.capitalize()} Model: R^2 Score: {r2:.2f}')
