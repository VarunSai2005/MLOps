"""FastAPI application for serving Titanic model predictions."""

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CLASS_LABELS = {
    0: "Did not survive",
    1: "Survived",
}


class PassengerRequest(BaseModel):
    """Validated passenger features expected by the trained pipeline."""

    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None


class PredictionResponse(BaseModel):
    """Response returned for a valid prediction request."""

    predicted_survived: int
    predicted_label: Literal["Did not survive", "Survived"]
    probability_did_not_survive: float | None = None
    probability_survived: float | None = None


def load_model(project_root: Path) -> Any:
    """Load the existing fitted pipeline; never train a replacement model."""
    model_path = project_root / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError("The trained model artifact model.pkl is missing.")
    return joblib.load(model_path)


PROJECT_ROOT = Path(__file__).resolve().parent
try:
    model = load_model(PROJECT_ROOT)
    model_load_error: Exception | None = None
except Exception as error:
    # Keep the app importable so health checks can report that it is not ready.
    model = None
    model_load_error = error
    logger.error("Unable to load model.pkl; API readiness is unavailable: %s", error)

app = FastAPI(
    title="Titanic ML Model Serving API",
    description="Serve predictions from the fitted Titanic classification pipeline.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, object]:
    """Report whether the API process has a usable model loaded."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable. Start the API after fixing the model artifact.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(passenger: PassengerRequest) -> PredictionResponse:
    """Predict survival for one validated passenger record."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable. Prediction cannot be served.",
        )

    # Keep the DataFrame columns exactly aligned with the training pipeline.
    passenger_frame = pd.DataFrame(
        [passenger.model_dump()],
        columns=FEATURES,
    )
    try:
        predicted_class = int(model.predict(passenger_frame)[0])
        response: dict[str, Any] = {
            "predicted_survived": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
        }
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(passenger_frame)[0]
            class_probabilities = {
                int(class_value): float(probabilities[index])
                for index, class_value in enumerate(model.classes_)
            }
            response["probability_did_not_survive"] = class_probabilities[0]
            response["probability_survived"] = class_probabilities[1]
        return PredictionResponse(**response)
    except (KeyError, TypeError, ValueError) as error:
        # Avoid exposing implementation details or local paths to API clients.
        raise HTTPException(
            status_code=500,
            detail="The model could not process this prediction request.",
        ) from error
