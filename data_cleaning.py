import pandas as pd

# Load the dataset
df = pd.read_csv("StudyFlow_10000_Students_Dataset-3.csv")

# Check dataset size
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

#check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

#check for duplicate rows
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:",duplicate_count)

#check data types
print("\nData types of each column:")
print(df.dtypes)

#convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"],errors="coerce")

#check if any invalid dates were created
print("\nInvalid dates:",df["Date"].isna().sum())

# Check valid ranges of important columns
print("\nInvalid Study Duration:",
      ((df["Study_Duration_Min"] < 0) |
       (df["Study_Duration_Min"] > 600)).sum())

print("Invalid Difficulty:",
      ((df["Difficulty_1_5"] < 1) |
       (df["Difficulty_1_5"] > 5)).sum())

print("Invalid Initial Scores:",
      ((df["Initial_Score"] < 0) |
       (df["Initial_Score"] > 100)).sum())

print("Invalid Revision Counts:",
      (df["Revision_Count"] < 0).sum())

print("Invalid Days Since Study:",
      (df["Days_Since_Study"] < 0).sum())

print("Invalid Later Scores:",
      ((df["Later_Score"] < 0) |
       (df["Later_Score"] > 100)).sum())

print("Invalid Focus Levels:",
      ((df["Focus_Level_1_5"] < 1) |
       (df["Focus_Level_1_5"] > 5)).sum())

print("Invalid Retention Scores:",
      ((df["Retention_Score"] < 0) |
       (df["Retention_Score"] > 100)).sum())

# Save cleaned dataset
output_file = "StudyFlow_Cleaned_Dataset.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

