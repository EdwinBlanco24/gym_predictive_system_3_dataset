from operator import index

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
    st.header("🏋️ Predicción de Adherencia")
    st.markdown("### Complete su perfil fitness")

    # -----------------------------
    # FORM LAYOUT
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Edad",
            min_value=18,
            max_value=80,
            key="age",
            value=None,
            placeholder="Ingrese su edad"
        )
        

        gender = st.selectbox(
            "Género",
            ["Masculino", "Femenino"],
            index=None,
            key="gender",
            placeholder="Seleccione su género"
        )

        weight = st.number_input(
            "Peso (kg)",
            min_value=30.0,
            max_value=200.0,
            value=None,
            key="weight",
            placeholder="Ingrese su peso en kg"
        )

        height = st.number_input(
            "Altura (Mts)",
            min_value=1.0,
            max_value=2.5,
            value=None,
            key="height",
            placeholder="Ingrese su altura en metros"
        )

        session_duration = st.number_input(
            "Duración (Horas) de la Sesión de Entrenamiento",
            min_value=0.5,
            max_value=5.0,
            value=None,
            key="session_duration",
            placeholder="Ingrese la duración de la sesión en horas"
        )

        calories = st.number_input(
            "Calorias Quemadas por Sesión (kcal)",
            min_value=100.0,
            max_value=3000.0,
            value=None,
            key="calories",
            placeholder="Ingrese las calorías quemadas por sesión"
        )

    with col2:
        fat_percentage = st.number_input(
            "Porcentaje de Grasa Corporal (%)",
            min_value=5.0,
            max_value=60.0,
            value=None,
            key="fat_percentage",
            placeholder="Ingrese su porcentaje de grasa corporal"
        )

        water = st.number_input(
            "Ingesta de Agua Promedio x Día (litros)",
            min_value=0.5,
            max_value=10.0,
            value=None,
            key="water",
            placeholder="Ingrese su ingesta de agua promedio por día"
        )

        frequency = st.number_input(
            "Frecuencia de Entrenamiento (días/semana)",
            min_value=1,
            max_value=14,
            value=None,
            key="frequency",
            placeholder="Ingrese la frecuencia de entrenamiento"
        )

        avg_bpm = st.number_input(
            "Promedio BPM (Frecuencia Cardíaca)",
            min_value=40,
            max_value=220,
            value=None,
            key="avg_bpm",
            placeholder="Ingrese su frecuencia cardíaca promedio"
        )

        max_bpm = st.number_input(
            "Máximo BPM (Frecuencia Cardíaca)",
            min_value=40,
            max_value=240,
            value=None,
            key="max_bpm",
            placeholder="Ingrese su frecuencia cardíaca máxima"
        )

        resting_bpm = st.number_input(
            "Mínimo BPM (Frecuencia Cardíaca)",
            min_value=30,
            max_value=120,
            value=None,
            key="resting_bpm",
            placeholder="Ingrese su frecuencia cardíaca en reposo"
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
        # FEATURE ENGINEERING
        # -----------------------------
        bmi = weight / (height ** 2)
        hydration_score = water * frequency
        heart_rate_reserve = max_bpm - resting_bpm
        calories_per_hour = calories / (session_duration + 0.01)

        gender_value = 1 if gender == "Masculino" else 0

        # Default workout type
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
        # EXACT TRAINING ORDER
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
        # SCALE
        # -----------------------------
        scaled_data = scaler.transform(input_data)

        # -----------------------------
        # PREDICT
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
            f"Nivel de Experiencia Predicho: {result}"
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
    # NEW ADHERENCE BUTTON
    # -----------------------------
    if st.button(
        "🔄 Realizar Nueva Adherencia", use_container_width=True):
            for key in form_fields.keys():
                if key in st.session_state:
                     del st.session_state[key]
            st.rerun()