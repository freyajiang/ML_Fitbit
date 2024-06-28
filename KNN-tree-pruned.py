import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('merged.csv')
numeric_df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(df.columns[[0, 1, 3]], axis=1)
target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

def evaluate_decision_tree(max_depth_values):
    results = {}
    for max_depth in max_depth_values:
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[max_depth] = accuracy

    return results

max_depth_values = range(1, 6)
accuracy_results = evaluate_decision_tree(max_depth_values)

for max_depth, accuracy in accuracy_results.items():
    print(f'max_depth = {max_depth}, Accuracy: {accuracy:.3f}')
