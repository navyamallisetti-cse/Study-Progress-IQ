import pandas as pd

# Load cleaned dataset
df = pd.read_csv("StudyFlow_Cleaned_Dataset.csv")

print("Dataset Shape:", df.shape)

print("\nBasic Statistical Summary:")
print(df.describe())

print("\nSubjects:")
print(df["Subject"].value_counts())

print("\nRetention Risk:")
print(df["Retention_Risk"].value_counts())

print("\nAverage Study Duration:")
print(df["Study_Duration_Min"].mean())

print("\nAverage Initial Score:")
print(df["Initial_Score"].mean())

print("\nAverage Later Score:")
print(df["Later_Score"].mean())

import matplotlib.pyplot as plt

# Subject-wise average later score
subject_scores = df.groupby("Subject")["Later_Score"].mean()

plt.figure(figsize=(8, 5))
subject_scores.plot(kind="bar")

plt.title("Average Later Score by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Later Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("subject_average_score.png")
plt.show()

# Revision count vs average later score
revision_scores = df.groupby("Revision_Count")["Later_Score"].mean()

plt.figure(figsize=(8, 5))
revision_scores.plot(kind="bar")

plt.title("Average Later Score by Revision Count")
plt.xlabel("Number of Revisions")
plt.ylabel("Average Later Score")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("revision_vs_score.png")
plt.show()

