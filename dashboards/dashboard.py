import streamlit as st
import pandas as pd
import plotly.express as px

from config.db import connect_db


def render_dashboard():
    st.header("📈 Dashboard Analitico Fitness con IA")
    st.markdown("📊 Inteligencia visual basada en registros predictivos del sistema")

    conn = connect_db()

    adherence = pd.read_sql("SELECT * FROM adherence_predictions", conn)
    injury = pd.read_sql("SELECT * FROM injury_risk_predictions", conn)

    # -----------------------------
    # KPI METRICS
    # -----------------------------
    total_adherence = len(adherence)
    total_injury = len(injury)

    advanced_users = len(
        adherence[
            adherence["prediction_result"].isin(["Avanzado", "Advanced"]
        )
        ]
    ) if not adherence.empty else 0

    high_risk_users = len(
        injury[
            injury["prediction_result"].isin(
            ["Alto Riesgo", "High Risk"]
            )
        ]
    ) if not injury.empty else 0

    conn.close()

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Predicciones de Adherencia", total_adherence)

    kpi2.metric("Predicciones de Riesgo de Lesión",total_injury)

    kpi3.metric("Usuarios Avanzados", advanced_users)

    kpi4.metric("Casos de Alto Riesgo", high_risk_users)

    # -----------------------------
    # ADHERENCE SECTION
    # -----------------------------
    st.subheader("🏋️ Predicciones de Nivel Fitness (Adherencia)")

    if not adherence.empty:

        # -----------------------------
        # TWO COLUMN VISUALS
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(
                adherence,
                x="prediction_result",
                title="Distribución de Nivel Fitness",
                template="plotly_white",
                color="prediction_result",
                labels={"prediction_result": "Resultado de Predicción"}
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

        with col2:
            fig2 = px.scatter(
                adherence,
                x="bmi",
                y="weekly_sessions",
                color="prediction_result",
                size="motivation_level",
                title="BMI vs Frecuencia de Entrenamiento",
                template="plotly_white",
                labels={"bmi": "Indice de Masa Corporal (BMI)", "weekly_sessions": "Sesiones Semanales", "motivation_level": "Nivel de Motivación", "prediction_result": "Nivel Fitness"}
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # -----------------------------
        # BOXPLOT
        # -----------------------------
        fig3 = px.box(
            adherence,
            x="prediction_result",
            y="bmi",
            color="prediction_result",
            title="Distribución de BMI (Indice Masa Corporal) por Nivel",
            template="plotly_white",
            labels={"prediction_result": "Nivel Fitness", "bmi": "Indice de Masa Corporal (BMI)"}
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # -----------------------------
        # DATA TABLE
        # -----------------------------
        st.markdown("### 📋 Historial de predicciones de nivel fitness")
        # -----------------------------
        # TRANSLATE ADHERENCE TABLE
        # -----------------------------
        adherence_display = adherence.rename(columns={
            "age": "Edad",
            "bmi": "BMI (Indice de Masa Corporal)",
            "weekly_sessions": "Sesiones semanales",
            "motivation_level": "Nivel de motivación",
            "experience_level": "Nivel previo",
            "prediction_result": "Resultado",
            "created_at": "Fecha de registro"
        })

        adherence_display["Resultado"] = adherence_display["Resultado"].replace({
            "Beginner": "Principiante",
            "Intermediate": "Intermedio",
            "Advanced": "Avanzado"
        })

        st.dataframe(adherence_display)


    # -----------------------------
    # INJURY RISK SECTION
    # -----------------------------
    st.subheader("⚠️ Predicciones de Riesgo de Lesión")

    if not injury.empty:

        # -----------------------------
        # TWO COLUMN VISUALS
        # -----------------------------
        col3, col4 = st.columns(2)

        with col3:
            fig4 = px.histogram(
                injury,
                x="prediction_result",
                title="Distribución de Riesgo de Sobreentrenamiento",
                template="plotly_white",
                color="prediction_result",
                labels={"prediction_result": "Resultado de Predicción"}
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

        with col4:
            fig5 = px.scatter(
                injury,
                x="training_load",
                y="recovery_score",
                color="prediction_result",
                title="Carga de Entrenamiento vs Recuperación",
                template="plotly_white",
                labels={"training_load": "Carga de Entrenamiento",
                         "recovery_score": "Puntuación de Recuperación", "prediction_result": "Riesgo de Lesión"}
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

        # -----------------------------
        # VIOLIN PLOT
        # -----------------------------
        fig6 = px.violin(
            injury,
            x="prediction_result",
            y="sleep_hours",
            color="prediction_result",
            box=True,
            title="Distribución del Sueño por Riesgo",
            template="plotly_white",
            labels={"prediction_result": "Riesgo de Lesión", "sleep_hours": "Horas de Sueño" }
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )

    # -----------------------------
    # DATA TABLE
    # -----------------------------
    st.markdown("### 📋 Historial de predicciones de lesiones")
    # -----------------------------
    # TRANSLATE INJURY TABLE
    # -----------------------------
    injury_display = injury.rename(columns={
        "age": "Edad",
        "bmi": "BMI",
        "fatigue_level": "Nivel de Fatiga",
        "training_load": "Carga de Entrenamiento",
        "recovery_score": "Puntuación de Recuperación",
        "sleep_hours": "Horas de Sueño",
        "prediction_result": "Resultado",
        "created_at": "Fecha de registro"
    })

    injury_display["Resultado"] = injury_display["Resultado"].replace({
        "Low Risk": "Bajo Riesgo",
        "High Risk": "Alto Riesgo"
    })

    st.dataframe(injury_display)

    # -----------------------------
    # GLOBAL SYSTEM SUMMARY
    # -----------------------------
    st.markdown("---")
    st.subheader("🧠 Resumen Global del Sistema Predictivo")
    summary_data = pd.DataFrame({
        "Métrica": [
            "Predicciones de Adherencia",
            "Predicciones de Lesiones",
            "Usuarios Avanzados",
            "Casos de Alto Riesgo",
            "Casos de Bajo Riesgo"
        ],
        "Valor": [
            total_adherence,
            total_injury,
            advanced_users,
            high_risk_users,
            total_injury - high_risk_users
        ]
    })

    fig7 = px.bar(
        summary_data,
        x="Métrica",
        y="Valor",
        color="Métrica",
        title="Resumen General del Sistema",
        template="plotly_white"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )