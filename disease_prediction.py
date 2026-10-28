
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Generate synthetic educational dataset
np.random.seed(42)

symptoms = [
    "fever",
    "cough",
    "headache",
    "fatigue",
    "sore_throat",
    "body_pain"
]

diseases = [
    "Flu-like Illness",
    "Common Cold",
    "Allergy"
]

rows = []

# Artificial patterns for demonstration only
for _ in range(300):
    disease = np.random.choice(diseases)

    if disease == "Flu-like Illness":
        values = [
            np.random.choice([0, 1], p=[0.2, 0.8]),
            np.random.choice([0, 1], p=[0.5, 0.5]),
            np.random.choice([0, 1], p=[0.4, 0.6]),
            np.random.choice([0, 1], p=[0.2, 0.8]),
            np.random.choice([0, 1], p=[0.5, 0.5]),
            np.random.choice([0, 1], p=[0.2, 0.8])
        ]

    elif disease == "Common Cold":
        values = [
            np.random.choice([0, 1], p=[0.7, 0.3]),
            np.random.choice([0, 1], p=[0.2, 0.8]),
            np.random.choice([0, 1], p=[0.7, 0.3]),
            np.random.choice([0, 1], p=[0.5, 0.5]),
            np.random.choice([0, 1], p=[0.3, 0.7]),
            np.random.choice([0, 1], p=[0.7, 0.3])
        ]

    else:
        values = [
            np.random.choice([0, 1], p=[0.9, 0.1]),
            np.random.choice([0, 1], p=[0.5, 0.5]),
            np.random.choice([0, 1], p=[0.6, 0.4]),
            np.random.choice([0, 1], p=[0.5, 0.5]),
            np.random.choice([0, 1], p=[0.8, 0.2]),
            np.random.choice([0, 1], p=[0.8, 0.2])
        ]

    rows.append(values + [disease])

df = pd.DataFrame(rows, columns=symptoms + ["disease"])

print("Dataset created successfully!")
print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)

# 2. Separate features and target
X = df[symptoms]
y = df["disease"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# 5. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Save dataset and model
df.to_csv("disease_dataset.csv", index=False)

import joblib
joblib.dump(model, "disease_prediction_model.pkl")

print("\nDataset saved as disease_dataset.csv")
print("Model saved as disease_prediction_model.pkl")

# 7. Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=ax,
    cmap="Blues"
)

ax.set_title("Disease Category Prediction - Confusion Matrix")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150)

print("Confusion matrix saved as confusion_matrix.png")

plt.show()
plt.close(fig)

# 8. Predict an example symptom pattern
new_patient = pd.DataFrame(
    [[1, 1, 1, 1, 0, 1]],
    columns=symptoms
)

prediction = model.predict(new_patient)[0]

print("\n========== EXAMPLE PREDICTION ==========")
print("Symptoms entered:")

for symptom, value in zip(symptoms, new_patient.iloc[0]):
    print(f"{symptom}: {'Yes' if value == 1 else 'No'}")

print("Predicted category:", prediction)

print("\nDisclaimer: Educational demonstration only.")
print("This model cannot diagnose real medical conditions.")
