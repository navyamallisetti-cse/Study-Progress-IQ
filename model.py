import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load cleaned dataset
df = pd.read_csv("StudyProgressIQ_Cleaned_Dataset.csv")

print("Dataset Shape:", df.shape)


# Features
X = df[
    [
        "Study_Duration_Min",
        "Difficulty_1_5",
        "Study_Method",
        "Initial_Score",
        "Revision_Count",
        "Days_Since_Study",
        "Focus_Level_1_5",
        "Subject",
        "Topic"
    ]
]

# Target
y = df["Retention_Risk"]


# Separate categorical and numerical columns
categorical_columns = [
    "Study_Method",
    "Subject",
    "Topic"
]

numerical_columns = [
    "Study_Duration_Min",
    "Difficulty_1_5",
    "Initial_Score",
    "Revision_Count",
    "Days_Since_Study",
    "Focus_Level_1_5"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Machine Learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# Split data by Student_ID
student_ids = df["Student_ID"].unique()

train_students, test_students = train_test_split(
    student_ids,
    test_size=0.20,
    random_state=42
)

train_df = df[df["Student_ID"].isin(train_students)]
test_df = df[df["Student_ID"].isin(test_students)]


X_train = train_df[X.columns]
y_train = train_df["Retention_Risk"]

X_test = test_df[X.columns]
y_test = test_df["Retention_Risk"]


# Train model
pipeline.fit(X_train, y_train)


# Prediction
y_pred = pipeline.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# CONFUSION MATRIX
# =========================================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("StudyProgressIQ - Retention Risk Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessing"]

feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_

feature_importance = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    .sort_values("Importance", ascending=False)
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))


# Save feature importance
feature_importance.head(15).to_csv(
    "feature_importance.csv",
    index=False
)


print("\nActual Risk Distribution:")
print(y_test.value_counts())

print("\nPredicted Risk Distribution:")
print(pd.Series(y_pred).value_counts())