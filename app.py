from flask import Flask, render_template, request
import pandas as pd
from datetime import date, timedelta

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)


# =========================================================
# LOAD CLEANED DATASET
# =========================================================

df = pd.read_csv("StudyProgressIQ_Cleaned_Dataset.csv")


# =========================================================
# DASHBOARD STATISTICS
# =========================================================

total_records = len(df)
total_students = df["Student_ID"].nunique()

avg_initial_score = round(df["Initial_Score"].mean(), 2)
avg_later_score = round(df["Later_Score"].mean(), 2)
avg_retention = round(df["Retention_Score"].mean(), 2)

risk_counts = df["Retention_Risk"].value_counts()

high_risk = int(risk_counts.get("High", 0))
medium_risk = int(risk_counts.get("Medium", 0))
low_risk = int(risk_counts.get("Low", 0))


# =========================================================
# GRAPH DATA
# =========================================================

# Average score by subject
subject_data = (
    df.groupby("Subject")["Later_Score"]
    .mean()
    .round(2)
    .to_dict()
)

# Average score by revision count
revision_data = (
    df.groupby("Revision_Count")["Later_Score"]
    .mean()
    .round(2)
    .to_dict()
)

# Risk distribution
risk_data = {
    "High": high_risk,
    "Medium": medium_risk,
    "Low": low_risk
}


# =========================================================
# MACHINE LEARNING FEATURES
# =========================================================

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


categorical_columns = [
    "Study_Method",
    "Subject",
    "Topic"
]


# =========================================================
# PREPROCESSING
# =========================================================

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


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# Train model
pipeline.fit(X, y)


# =========================================================
# HOME PAGE
# =========================================================


@app.route("/")
def home():

    return render_template(
        "index.html",

        total_records=total_records,
        total_students=total_students,

        avg_initial_score=avg_initial_score,
        avg_later_score=avg_later_score,
        avg_retention=avg_retention,

        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk,

        subject_data=subject_data,
        revision_data=revision_data,
        risk_data=risk_data
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # GET USER INPUT
    # -----------------------------------------------------

    study_duration = float(
        request.form["study_duration"]
    )

    difficulty = int(
        request.form["difficulty"]
    )

    study_method = request.form["study_method"]

    initial_score = float(
        request.form["initial_score"]
    )

    revision_count = int(
        request.form["revision_count"]
    )

    days_since_study = int(
        request.form["days_since_study"]
    )

    focus_level = int(
        request.form["focus_level"]
    )

    subject = request.form["subject"]

    topic = request.form["topic"]

    # Available daily study time
    available_time = int(
        request.form.get("available_time", study_duration)
    )


    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    student = pd.DataFrame([{

        "Study_Duration_Min": study_duration,

        "Difficulty_1_5": difficulty,

        "Study_Method": study_method,

        "Initial_Score": initial_score,

        "Revision_Count": revision_count,

        "Days_Since_Study": days_since_study,

        "Focus_Level_1_5": focus_level,

        "Subject": subject,

        "Topic": topic

    }])


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = pipeline.predict(student)[0]


    probabilities = pipeline.predict_proba(student)[0]

    confidence = max(probabilities) * 100


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    if prediction == "High":

        recommendation = (
            "Revise this topic soon, increase practice, "
            "and use active learning methods."
        )

    elif prediction == "Medium":

        recommendation = (
            "Schedule another revision and practice "
            "questions to strengthen retention."
        )

    else:

        recommendation = (
            "Your current learning pattern is good. "
            "Continue regular revision."
        )


    # =====================================================
    # ACTION PLAN
    # =====================================================

    action_plan = []

    if prediction == "High":

        action_plan = [
            "Revise the topic today.",
            "Use active recall instead of only reading.",
            "Practice 5–10 questions from this topic.",
            "Use flashcards or short self-tests.",
            "Check your score again after the next revision."
        ]

    elif prediction == "Medium":

        action_plan = [
            "Plan the next revision within 1–2 days.",
            "Practice at least 5 questions.",
            "Write a short summary of important concepts.",
            "Use active recall during revision.",
            "Track your next performance."
        ]

    else:

        action_plan = [
            "Continue your current study pattern.",
            "Practice a few questions regularly.",
            "Review important concepts periodically.",
            "Maintain consistent study sessions.",
            "Monitor your future retention."
        ]


    # =====================================================
    # STUDY METHOD ACTION
    # =====================================================

    if study_method in ["Reading", "Video"]:

        action_plan.append(
            "Add practice or flashcards to make learning more active."
        )

    else:

        action_plan.append(
            "Continue using active learning and practice-based methods."
        )


    # =====================================================
    # PERSONALIZED TIMETABLE
    # =====================================================

    timetable = []

    total_time = max(30, available_time)

    revision_time = round(total_time * 0.30)
    practice_time = round(total_time * 0.40)
    review_time = total_time - revision_time - practice_time

    timetable.append({
        "activity": "Topic Revision",
        "duration": revision_time,
        "description": "Review important concepts and notes."
    })

    timetable.append({
        "activity": "Practice",
        "duration": practice_time,
        "description": "Solve questions and test your understanding."
    })

    timetable.append({
        "activity": "Quick Review",
        "duration": review_time,
        "description": "Use active recall and summarize what you learned."
    })


    # =====================================================
    # REVISION SCHEDULE
    # =====================================================

    today = date.today()

    if prediction == "High":

        revision_days = [0, 1, 3]

    elif prediction == "Medium":

        revision_days = [1, 3, 7]

    else:

        revision_days = [3, 7, 14]


    revision_schedule = []

    for i, day_offset in enumerate(revision_days):

        revision_date = today + timedelta(days=day_offset)

        revision_schedule.append({
            "session": f"Revision {i + 1}",
            "date": revision_date.strftime("%d-%m-%Y"),
            "day": revision_date.strftime("%A")
        })


    # =====================================================
    # TOPIC INSIGHT
    # =====================================================

    if initial_score < 50:

        topic_insight = (
            "This topic needs stronger understanding. "
            "Start with basic concepts before advanced practice."
        )

    elif initial_score < 75:

        topic_insight = (
            "This topic has moderate performance. "
            "Regular revision and practice can strengthen it."
        )

    else:

        topic_insight = (
            "Your initial performance is relatively strong. "
            "Focus on maintaining retention through periodic revision."
        )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return render_template(

        "index.html",

        total_records=total_records,
        total_students=total_students,

        avg_initial_score=avg_initial_score,
        avg_later_score=avg_later_score,
        avg_retention=avg_retention,

        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk,

        subject_data=subject_data,
        revision_data=revision_data,
        risk_data=risk_data,

        prediction=prediction,
        confidence=round(confidence, 2),
        recommendation=recommendation,

        action_plan=action_plan,
        timetable=timetable,
        revision_schedule=revision_schedule,
        topic_insight=topic_insight

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)