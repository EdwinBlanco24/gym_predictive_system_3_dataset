import streamlit as st
import pandas as pd
import joblib

from config.db import save_adherence

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = joblib.load("models/adherence_model.pkl")
scaler = joblib.load("models/adherence_scaler.pkl")


def render_adherence_form():

    # -----------------------------
    # INITIALIZE FORM STATE
    # -----------------------------
    form_fields = {
        "age": None,
        "gender": None,
        "weight": None,
        "height": None,
        "session_duration": None,
        "calories": None,
        "fat_percentage": None,
        "water": None,
        "frequency": None,
        "avg_bpm": None,
        "max_bpm": None,
        "resting_bpm": None
    }

    for key, value in form_fields.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -----------------------------
    # HEADER
    # -----------------------------
    st.header("🏋️ Predicción de Adherencia")
    st.markdown("### Complete su perfil fitness")

    # -----------------------------
    # FORM LAYOUT
    # -----------------------------
    col1, col2 = st.columns(2)

    # -----------------------------
    # COLUMN 1
    # -----------------------------
    with col1:

        age = st.number_input(
            "Edad",
            min_value=18,
            max_value=80,
            value=None,
            placeholder="Ingrese su edad",
            key="age"
        )

        gender = st.selectbox(
            "Género",
            ["Masculino", "Femenino"],
            index=None,
            placeholder="Seleccione su género",
            key="gender"
        )

        weight = st.number_input(
            "Peso (kg)",
            min_value=30.0,
            max_value=200.0,
            value=None,
            placeholder="Ej: 75.5",
            key="weight"
        )

        height = st.number_input(
            "Altura (mts)",
            min_value=1.0,
            max_value=2.5,
            value=None,
            placeholder="Ej: 1.75",
            key="height"
        )

        session_duration = st.number_input(
            "Duración de la sesión (horas)",
            min_value=0.5,
            max_value=5.0,
            value=None,
            placeholder="Ej: 1.5",
            key="session_duration"
        )

        calories = st.number_input(
            "Calorías quemadas por sesión",
            min_value=100.0,
            max_value=3000.0,
            value=None,
            placeholder="Ej: 650",
            key="calories"
        )

    # -----------------------------
    # COLUMN 2
    # -----------------------------
    with col2:

        fat_percentage = st.number_input(
            "Porcentaje de grasa corporal (%)",
            min_value=5.0,
            max_value=60.0,
            value=None,
            placeholder="Ej: 18",
            key="fat_percentage"
        )

        water = st.number_input(
            "Ingesta de agua diaria (litros)",
            min_value=0.5,
            max_value=10.0,
            value=None,
            placeholder="Ej: 2.5",
            key="water"
        )

        frequency = st.number_input(
            "Frecuencia de entrenamiento (días/semana)",
            min_value=1,
            max_value=14,
            value=None,
            placeholder="Ej: 5",
            key="frequency"
        )

        avg_bpm = st.number_input(
            "Promedio BPM",
            min_value=40,
            max_value=220,
            value=None,
            placeholder="Ej: 130",
            key="avg_bpm"
        )

        max_bpm = st.number_input(
            "Máximo BPM",
            min_value=40,
            max_value=240,
            value=None,
            placeholder="Ej: 185",
            key="max_bpm"
        )

        resting_bpm = st.number_input(
            "BPM en reposo",
            min_value=30,
            max_value=120,
            value=None,
            placeholder="Ej: 65",
            key="resting_bpm"
        )

    st.markdown("---")

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------
    if st.button(
        "🚀 Predecir Adherencia",
        use_container_width=True
    ):

        # -----------------------------
        # VALIDATE REQUIRED FIELDS
        # -----------------------------
        required_fields = [
            age,
            gender,
            weight,
            height,
            session_duration,
            calories,
            fat_percentage,
            water,
            frequency,
            avg_bpm,
            max_bpm,
            resting_bpm
        ]

        if any(field is None for field in required_fields):
            st.warning(
                "⚠️ Complete todos los campos antes de realizar la predicción."
            )
            st.stop()

        # -----------------------------
        # CAST TYPES
        # -----------------------------
        age = int(age or 0)
        weight = float(weight or 0)
        height = float(height or 0)
        session_duration = float(session_duration or 0)
        calories = float(calories or 0)
        fat_percentage = float(fat_percentage or 0)
        water = float(water or 0)
        frequency = int(frequency or 0)
        avg_bpm = int(avg_bpm or 0)
        max_bpm = int(max_bpm or 0)
        resting_bpm = int(resting_bpm or 0 )

        # -----------------------------
        # FEATURE ENGINEERING
        # -----------------------------
        bmi = weight / (height ** 2)

        hydration_score = water * frequency

        heart_rate_reserve = max_bpm - resting_bpm

        calories_per_hour = calories / (
            session_duration + 0.01
        )

        gender_value = 1 if gender == "Masculino" else 0

        # -----------------------------
        # DEFAULT WORKOUT TYPES
        # -----------------------------
        workout_hiit = 0
        workout_strength = 1
        workout_yoga = 0

        # -----------------------------
        # INPUT DATA
        # -----------------------------
        input_dict = {
            "Age": age,
            "Gender": gender_value,
            "Weight_kg": weight,
            "Height_m": height,
            "Max_BPM": max_bpm,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration_hours": session_duration,
            "Calories_Burned": calories,
            "Fat_Percentage": fat_percentage,
            "Water_Intake_liters": water,
            "Workout_Frequency": frequency,
            "BMI": bmi,
            "Workout_Type_HIIT": workout_hiit,
            "Workout_Type_Strength": workout_strength,
            "Workout_Type_Yoga": workout_yoga,
            "Heart_Rate_Reserve": heart_rate_reserve,
            "Calories_per_Hour": calories_per_hour,
            "Hydration_Score": hydration_score
        }

        # -----------------------------
        # FEATURE ORDER
        # -----------------------------
        feature_order = [
            "Age",
            "Gender",
            "Weight_kg",
            "Height_m",
            "Max_BPM",
            "Avg_BPM",
            "Resting_BPM",
            "Session_Duration_hours",
            "Calories_Burned",
            "Fat_Percentage",
            "Water_Intake_liters",
            "Workout_Frequency",
            "BMI",
            "Workout_Type_HIIT",
            "Workout_Type_Strength",
            "Workout_Type_Yoga",
            "Heart_Rate_Reserve",
            "Calories_per_Hour",
            "Hydration_Score"
        ]

        input_data = pd.DataFrame(
            [[input_dict[col] for col in feature_order]],
            columns=feature_order
        )

        # -----------------------------
        # SCALE DATA
        # -----------------------------
        scaled_data = scaler.transform(input_data)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        prediction = model.predict(scaled_data)[0]

        # -----------------------------
        # MAP RESULT
        # -----------------------------
        level_map = {
            1: "Principiante",
            2: "Intermedio",
            3: "Avanzado"
        }

        result = level_map.get(
            prediction,
            str(prediction)
        )

        # -----------------------------
        # SHOW RESULT
        # -----------------------------
        st.success(
            f"🏋️ Nivel de experiencia predicho: {result}"
        )

        # -----------------------------
        # SAVE TO DATABASE
        # -----------------------------
        save_adherence(
            age,
            gender,
            bmi,
            frequency,
            session_duration,
            fat_percentage,
            result
        )

    # -----------------------------
    # RESET FORM
    # -----------------------------
    if st.button(
        "🔄 Realizar Nueva Adherencia",
        use_container_width=True
    ):

        for key in form_fields.keys():
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()