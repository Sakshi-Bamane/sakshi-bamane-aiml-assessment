from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="Lead Conversion Prediction API"
)

# Load saved model
model_data = joblib.load("model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]


@app.get("/")
def home():
    return {
        "status": "API Running",
        "model": "Random Forest"
    }


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Keep same feature order
    df = df[features]

    # Scale
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "conversion_probability": round(float(probability), 4)
    }

# ----------------------------------
# Explain Endpoint
# ----------------------------------

@app.post("/explain")
def explain(data: dict):

    probability = data["conversion_probability"]

    if probability > 0.8:
        summary = (
            "High conversion likelihood. Lead has strong engagement signals "
            "and should be prioritized by the sales team."
        )

    elif probability > 0.5:
        summary = (
            "Moderate conversion likelihood. Additional nurturing campaigns "
            "may improve conversion chances."
        )

    else:
        summary = (
            "Low conversion likelihood. Lead currently shows limited engagement "
            "and may require re-targeting."
        )

    return {
        "summary": summary
    }