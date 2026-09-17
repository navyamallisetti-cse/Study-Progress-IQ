# Study Progress IQ

### Learning Retention & Performance Analytics System

Study Progress IQ is a data-driven learning analytics system that analyzes study patterns and learning performance to predict retention risk and provide personalized study recommendations.

## 🎯 Project Overview

The system follows:

**Analyze → Predict → Take Action → Track Improvement**

It helps learners:

- Analyze their learning patterns
- Identify topics that need more attention
- Predict learning retention risk
- Receive personalized study recommendations
- Generate a daily study timetable
- Create a revision schedule
- Track learning performance

## 🚀 Key Features

### 📊 Learning Dashboard
Provides an overview of learning performance and retention statistics.

### 🧠 Learning Pattern Analysis
Analyzes study duration, difficulty, study method, scores, revisions, focus level, subject, and topic.

### 🎯 Retention Risk Prediction
Uses Machine Learning to classify the predicted retention risk as:

- Low
- Medium
- High

### 💡 Personalized Recommendations
Provides recommendations based on the predicted retention risk and learning pattern.

### 📅 Personalized Study Timetable
Generates a study plan based on the learner's available study time.

### 🔄 Revision Schedule
Creates a personalized revision schedule to support regular review.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Flask
- HTML
- CSS
- Machine Learning

## 🤖 Machine Learning

A Random Forest Classifier is used for retention-risk prediction.

The model uses learning-related features such as:

- Study Duration
- Difficulty Level
- Study Method
- Initial Score
- Revision Count
- Days Since Study
- Focus Level
- Subject
- Topic

## 📂 Project Structure

```text
Study-Progress-IQ/
│
├── app.py
├── data_cleaning.py
├── eda.py
├── model.py
├── predict.py
│
├── StudyFlow_10000_Students_Dataset-3.csv
├── StudyFlow_Cleaned_Dataset.csv
├── feature_importance.csv
│
├── subject_average_score.png
├── revision_vs_score.png
│
├── static/
│   ├── style.css
│   └── confusion_matrix.png
│
└── templates/
    ├── index.html
    ├── analytics.html
    ├── analyze.html
    ├── planner.html
    ├── revision.html
    └── summary.html