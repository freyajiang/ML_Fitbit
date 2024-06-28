# probably not applicable since this normally requires binary data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv('merged.csv')
df = df.drop(df.columns[[0, 1, 3]], axis=1)  # dropped columns 0 (id), 1 (date), 3 (time in bed)
for column in df.columns:
    median = df[column].median()
    df[column] = (df[column] > median).astype(int)

target = 'TotalMinutesAsleep'
features = df.drop(columns=[target])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.3f}")

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.3f}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")

print("Trained model's formula:")
print(f"{target} = {' + '.join([f'{coef:3f} * {feat}' for coef, feat in zip(model.coef_[0], features.columns)])} + {model.intercept_[0]}")

