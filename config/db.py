import psycopg2


# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def connect_db():
    return psycopg2.connect(
        host="127.0.0.1",
        database="gym_predictive_system",
        user="postgres",
        password="9518746230.",
        port="5432"
    )


# -----------------------------
# SAVE ADHERENCE PREDICTION
# -----------------------------
def save_adherence(
    age,
    gender,
    bmi,
    workout_frequency,
    session_duration,
    fat_percentage,
    prediction_result
):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO adherence_predictions (
            age,
            bmi,
            weekly_sessions,
            motivation_level,
            experience_level,
            prediction_result
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        age,
        bmi,
        workout_frequency,
        session_duration,
        prediction_result,
        prediction_result
    ))

    conn.commit()
    cursor.close()
    conn.close()


# -----------------------------
# SAVE INJURY RISK
# -----------------------------
def save_injury_risk(
    age,
    training_hours,
    rest_hours,
    sleep_score,
    recovery_balance,
    prediction_result
):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO injury_risk_predictions (
            age,
            fatigue_level,
            sleep_hours,
            training_load,
            recovery_score,
            prediction_result
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        age,
        training_hours,
        sleep_score,
        training_hours,
        recovery_balance,
        prediction_result
    ))

    conn.commit()
    cursor.close()
    conn.close()