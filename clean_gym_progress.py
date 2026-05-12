import pandas as pd

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("datasets/Gym_Progress_Dataset.csv")

# -----------------------------
# DATE CONVERSION
# -----------------------------
df["Day"] = pd.to_datetime(df["Day"])

# -----------------------------
# TIME FEATURES
# -----------------------------
df["day_number"] = (df["Day"] - df["Day"].min()).dt.days
df["week"] = df["day_number"] // 7

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["Calories_per_Workout"] = (
    df["Calories_Intake"] /
    (df["Workout_Duration_min"] + 1)
)

df["Protein_per_kg"] = (
    df["Protein_Intake_g"] /
    df["Weight_kg"]
)

df["Activity_Score"] = (
    df["Workout_Duration_min"] +
    (df["Steps_Walked"] / 1000)
)

# -----------------------------
# WEIGHT CHANGE
# -----------------------------
df["Weight_Change"] = df["Weight_kg"].diff()

# -----------------------------
# BALANCED CLASSIFICATION TARGET
# -----------------------------
def classify_progress(change):
    if pd.isnull(change):
        return "maintenance"
    elif change < -0.8:
        return "weight_loss"
    elif change > 0.8:
        return "muscle_gain"
    else:
        return "maintenance"


df["Progress_Category"] = df["Weight_Change"].apply(
    classify_progress
)

# -----------------------------
# DROP UNUSED COLUMNS
# -----------------------------
df.drop(
    ["Day", "Weight_Change"],
    axis=1,
    inplace=True
)

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
print("Duplicates:", df.duplicated().sum())

# -----------------------------
# CLASS DISTRIBUTION
# -----------------------------
print("\nProgress Category Distribution:")
print(df["Progress_Category"].value_counts())

# -----------------------------
# SAVE CLEANED DATASET
# -----------------------------
df.to_csv(
    "datasets/cleaned_gym_progress.csv",
    index=False
)

print("\n✅ Gym Progress Dataset cleaned for balanced classification successfully")