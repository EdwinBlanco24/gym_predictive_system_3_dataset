from config.db import save_body_transformation

save_body_transformation(
    age=25,
    weight=75,
    height=178,
    body_fat=18,
    workout_frequency=4,
    goal="muscle_gain",
    prediction_result="Estimated muscle gain in 12 weeks"
)

print("✅ Record saved successfully")