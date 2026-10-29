
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the model saved by train_model.py
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "food_waste_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Food Waste Prediction API",
    description="Practice API for machine-learning food waste prediction",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    meals_served: int = Field(ge=0)
    kitchen_staff: int = Field(ge=0)
    temperature_C: float
    humidity_percent: float = Field(ge=0, le=100)
    day_of_week: int = Field(ge=0, le=6)
    special_event: int = Field(ge=0, le=1)
    past_waste_kg: float = Field(ge=0)
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    staff_experience: str
    waste_category: str


@app.get("/")
def home():
    return {
        "message": "Food Waste Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        values = data.model_dump()

        # Match the category formatting used during training
        values["staff_experience"] = (
            values["staff_experience"].strip().title()
        )
        values["waste_category"] = (
            values["waste_category"].strip().title()
        )

        allowed_experience = {
            "Beginner", "Intermediate", "Expert", "Unknown"
        }
        allowed_categories = {
            "Meat", "Dairy", "Vegetables", "Grains"
        }

        if values["staff_experience"] not in allowed_experience:
            raise HTTPException(
                status_code=422,
                detail="Use Beginner, Intermediate, Expert, or Unknown."
            )

        if values["waste_category"] not in allowed_categories:
            raise HTTPException(
                status_code=422,
                detail="Use Meat, Dairy, Vegetables, or Grains."
            )

        input_df = pd.DataFrame([values])

        prediction = float(model.predict(input_df)[0])

        return {
            "predicted_food_waste_kg": round(prediction, 2),
            "unit": "kg",
            "notice": (
                "Practice prediction using the current training dataset; "
                "not validated for wholesale-market use."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs."
        ) from exc
