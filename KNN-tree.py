import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

def evaluate_knn(k_values):
    results = {}
    for k in k_values:
        # Initialize KNN model
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the model
        knn.fit(X_train, y_train)

        # Predict on the test set
        y_pred = knn.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        results[k] = accuracy

    return results

k_values = range(1, 6)
accuracy_results = evaluate_knn(k_values)

for k, accuracy in accuracy_results.items():
    print(f'k = {k}, Accuracy: {accuracy:.3f}')
