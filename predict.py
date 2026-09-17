import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("StudyFlow_Cleaned_Dataset.csv")


# Features
features = [
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

X = df[features]
y = df["Retention_Risk"]


# Categorical columns
categorical_columns = [
    "Study_Method",
    "Subject",
    "Topic"
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


# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# Train model using complete dataset
pipeline.fit(X, y)


# Example student input
student = pd.DataFrame([{
    "Study_Duration_Min": 30,
    "Difficulty_1_5": 4,
    "Study_Method": "Reading",
    "Initial_Score": 60,
    "Revision_Count": 0,
    "Days_Since_Study": 7,
    "Focus_Level_1_5": 2,
    "Subject": "Python",
    "Topic": "Functions"
}])


# Prediction
prediction = pipeline.predict(student)[0]


print("\n===== STUDYFLOW RESULT =====")
print("Subject:", student["Subject"].iloc[0])
print("Topic:", student["Topic"].iloc[0])
print("Predicted Retention Risk:", prediction)


# Recommendation
if prediction == "High":
    print("Recommendation: Revise this topic soon and increase practice.")
elif prediction == "Medium":
    print("Recommendation: Schedule another revision and practice questions.")
else:
    print("Recommendation: Current learning pattern is good. Continue regular revision.")