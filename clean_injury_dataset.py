import pandas as pd

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("datasets/data.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Recovery balance
df["Recovery_Balance"] = (
    df["RestHoursPerWeek"] /
    (df["TrainingHoursPerWeek"] + 1)
)

# Stress index
df["Stress_Index"] = (
    df["CortisolLevel"] /
    (df["SleepQualityScore"] + 1)
)

# Cardiovascular strain
df["Cardio_Strain"] = (
    df["HeartRateRest"] /
    (df["HeartRateVar"] + 1)
)

# Hormonal balance
df["Hormonal_Balance"] = (
    df["TestosteroneLevel"] /
    (df["CortisolLevel"] + 1)
)

# Load stress
df["Load_Stress"] = (
    df["TrainingIntensity"] *
    df["TrainingLoadVariation"]
)

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
print("Duplicates:", df.duplicated().sum())

# -----------------------------
# TARGET DISTRIBUTION
# -----------------------------
print("\nOvertraining Distribution:")
print(df["Overtraining"].value_counts())

# -----------------------------
# SAVE CLEANED DATASET
# -----------------------------
df.to_csv(
    "datasets/cleaned_injury_dataset.csv",
    index=False
)

print("\n✅ Injury Dataset cleaned successfully")