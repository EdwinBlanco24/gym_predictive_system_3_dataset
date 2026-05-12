import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# -----------------------------
# LOAD CLEANED DATASET
# -----------------------------
df = pd.read_csv(
    "datasets/cleaned_gym_members.csv"
)

# -----------------------------
# FEATURES / TARGET
# -----------------------------
X = df.drop(
    ["Experience_Level"],
    axis=1
)

y = df["Experience_Level"]

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# PREDICT
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions
))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(
    model,
    "models/adherence_model.pkl"
)

joblib.dump(
    scaler,
    "models/adherence_scaler.pkl"
)

print("\n✅ Adherence Model trained successfully")