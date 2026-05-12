import pandas as pd

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(
    "datasets/gym_members_exercise_tracking.csv"
)

# -----------------------------
# RENAME COLUMNS
# -----------------------------
df.rename(columns={
    "Weight (kg)": "Weight_kg",
    "Height (m)": "Height_m",
    "Session_Duration (hours)": "Session_Duration_hours",
    "Water_Intake (liters)": "Water_Intake_liters",
    "Workout_Frequency (days/week)": "Workout_Frequency"
}, inplace=True)

# -----------------------------
# ENCODE GENDER
# -----------------------------
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

# -----------------------------
# ONE HOT ENCODE WORKOUT TYPE
# -----------------------------
df = pd.get_dummies(
    df,
    columns=["Workout_Type"],
    drop_first=True
)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["Heart_Rate_Reserve"] = (
    df["Max_BPM"] - df["Resting_BPM"]
)

df["Calories_per_Hour"] = (
    df["Calories_Burned"] /
    (df["Session_Duration_hours"] + 0.01)
)

df["Hydration_Score"] = (
    df["Water_Intake_liters"] *
    df["Workout_Frequency"]
)

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
print("Duplicates:", df.duplicated().sum())

# -----------------------------
# TARGET DISTRIBUTION
# -----------------------------
print("\nExperience Level Distribution:")
print(df["Experience_Level"].value_counts())

# -----------------------------
# SAVE CLEANED DATASET
# -----------------------------
df.to_csv(
    "datasets/cleaned_gym_members.csv",
    index=False
)

print("\n✅ Gym Members Dataset cleaned successfully")