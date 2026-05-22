from unittest import result

import streamlit as st
import pandas as pd
import joblib

from config.db import save_injury_risk

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/injury_model.pkl")
scaler = joblib.load("models/injury_scaler.pkl")


def render_injury_form():
    st.header("⚠️ Predicción de Riesgo de Lesión")
    st.markdown("### Ingrese sus métricas de recuperación y rendimiento")

        # -----------------------------
    # INITIALIZE FORM STATE
    # -----------------------------
    form_fields = {
        "injury_age": None,
        "training_hours": None,
        "training_intensity": None,
        "rest_hours": None,
        "sleep_score": None,
        "cortisol": None,
        "testosterone": None,
        "hr_rest": None,
        "hr_var": None,
        "nutrition": None
    }

    for key, value in form_fields.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -----------------------------
    # FORM LAYOUT
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", min_value=18, max_value=80, value=None, placeholder="Ingrese su edad", key="injury_age")
        training_hours = st.number_input("Horas de Entrenamiento Por Semana", min_value=1.0, max_value=60.0, placeholder="Ingrese las horas de entrenamiento", key="training_hours")
        training_intensity = st.number_input("Intensidad del Entrenamiento", min_value=1.0, max_value=10.0, placeholder="Ingrese la intensidad del entrenamiento", key="training_intensity")
        rest_hours = st.number_input("Horas de Descanso Por Semana", min_value=1.0, max_value=100.0, placeholder="Ingrese las horas de descanso", key="rest_hours")
        sleep_score = st.number_input("Puntuación de Calidad del Sueño", min_value=1.0, max_value=10.0, placeholder="Ingrese la puntuación de calidad del sueño", key="sleep_score")

    with col2:
        cortisol = st.number_input("Nivel de Cortisol", min_value=1.0, max_value=50.0, placeholder="Ingrese el nivel de cortisol", key="cortisol")
        testosterone = st.number_input("Nivel de Testosterona", min_value=1.0, max_value=50.0, placeholder="Ingrese el nivel de testosterona", key="testosterone")
        hr_rest = st.number_input("Frecuencia Cardíaca en Reposo", min_value=30.0, max_value=120.0, placeholder="Ingrese la frecuencia cardíaca en reposo", key="hr_rest")
        hr_var = st.number_input("Variabilidad de la Frecuencia Cardíaca", min_value=1.0, max_value=150.0, placeholder="Ingrese la variabilidad de la frecuencia cardíaca", key="hr_var")
        nutrition = st.number_input("Puntuación de Nutrición", min_value=1.0, max_value=10.0, placeholder="Ingrese la puntuación de nutrición", key="nutrition")

    st.markdown("---")

    # -----------------------------
    # PREDICTION BUTTON
    # -----------------------------
    if st.button("🛡️ Predecir Riesgo de Lesión", use_container_width=True):

        # -----------------------------
        # VALIDATE REQUIRED FIELDS
        # -----------------------------
        required_fields = [
            age,
            training_hours,
            training_intensity,
            rest_hours,
            sleep_score,
            hr_rest,
            hr_var,
            cortisol,
            testosterone,
            nutrition
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

        training_hours = float(training_hours or 0)

        training_intensity = float(training_intensity or 0)

        rest_hours = float(rest_hours or 0)

        sleep_score = float(sleep_score or 0)

        hr_rest = float(hr_rest or 0)

        hr_var = float(hr_var or 0)

        cortisol = float(cortisol or 0)

        testosterone = float(testosterone or 0)

        nutrition = float(nutrition or 0)

        recovery_balance = rest_hours / (training_hours + 1)
        stress_index = cortisol / (sleep_score + 1)
        cardio_strain = hr_rest / (hr_var + 1)
        hormonal_balance = testosterone / (cortisol + 1)
        load_stress = training_intensity * training_hours

        input_data = pd.DataFrame([{
            "Age": age,
            "TrainingHoursPerWeek": training_hours,
            "TrainingIntensity": training_intensity,
            "RestHoursPerWeek": rest_hours,
            "SleepQualityScore": sleep_score,
            "HeartRateRest": hr_rest,
            "HeartRateVar": hr_var,
            "CortisolLevel": cortisol,
            "TestosteroneLevel": testosterone,
            "CKLevel": 200,
            "MoodScore": 5,
            "ReactionTime": 200,
            "TrainingLoadVariation": 20,
            "InjuryHistory": 0,
            "NutritionScore": nutrition,
            "Recovery_Balance": recovery_balance,
            "Stress_Index": stress_index,
            "Cardio_Strain": cardio_strain,
            "Hormonal_Balance": hormonal_balance,
            "Load_Stress": load_stress
        }])

        # Scale
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]

        result = "Alto Riesgo" if prediction == 1 else "Bajo Riesgo"

        if prediction == 1:
            st.error(f"⚠️ Riesgo de sobreentrenamiento: {result}")
        else:
            st.success(f"✅ Riesgo de sobreentrenamiento: {result}")

        # Save to DB
        save_injury_risk(
            age,
            training_hours,
            rest_hours,
            sleep_score,
            recovery_balance,
            result
        )

    # -----------------------------
    # RESET FORM
    # -----------------------------
    if st.button("🔄 Realizar Nueva Predicción de Riesgo", use_container_width=True):
        for key in form_fields.keys():
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()