"""Simulate streaming inference for individual passenger records."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import time


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CLASS_LABELS = {
    0: "Did not survive",
    1: "Survived",
}


def load_model(project_root: Path) -> Any:
    """Load the existing fitted pipeline without retraining it."""
    model_path = project_root / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run train_model.py before running streaming inference."
        )
    return joblib.load(model_path)


def build_stream_records() -> list[dict[str, Any]]:
    """Create 10 new records that simulate incoming stream events."""
    # These illustrative events are distinct from train.csv and prior examples.
    return [
        {"Pclass": 1, "Sex": "female", "Age": 62.0, "SibSp": 0, "Parch": 1, "Fare": 142.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male", "Age": 33.0, "SibSp": 2, "Parch": 2, "Fare": 12.25, "Embarked": "Q"},
        {"Pclass": 2, "Sex": "female", "Age": 16.0, "SibSp": 1, "Parch": 0, "Fare": 20.0, "Embarked": "S"},
        {"Pclass": 1, "Sex": "male", "Age": 50.0, "SibSp": 0, "Parch": 2, "Fare": 88.0, "Embarked": "C"},
        {"Pclass": 3, "Sex": "female", "Age": 36.0, "SibSp": 3, "Parch": 0, "Fare": 16.75, "Embarked": "S"},
        {"Pclass": 2, "Sex": "male", "Age": None, "SibSp": 0, "Parch": 1, "Fare": 30.0, "Embarked": "C"},
        {"Pclass": 3, "Sex": "male", "Age": 5.0, "SibSp": 1, "Parch": 3, "Fare": 19.0, "Embarked": "S"},
        {"Pclass": 1, "Sex": "female", "Age": 44.0, "SibSp": 2, "Parch": 0, "Fare": 105.0, "Embarked": "Q"},
        {"Pclass": 2, "Sex": "female", "Age": 68.0, "SibSp": 0, "Parch": 0, "Fare": 31.0, "Embarked": None},
        {"Pclass": 3, "Sex": "male", "Age": 26.0, "SibSp": 0, "Parch": 4, "Fare": 22.5, "Embarked": "C"},
    ]


def main() -> None:
    """Load the model once and process each incoming record in sequence."""
    project_root = Path(__file__).resolve().parent
    model = load_model(project_root)
    stream_records = build_stream_records()

    print("Starting streaming inference simulation for 10 records.")
    for record_number, record in enumerate(stream_records, start=1):
        print(f"\nRecord {record_number} arrived.")

        # A stream event becomes a one-row DataFrame before entering the pipeline.
        passenger = pd.DataFrame([record], columns=FEATURES)
        prediction = int(model.predict(passenger)[0])

        print("Passenger features:")
        print(passenger.to_string(index=False))
        print(f"Predicted class: {prediction}")
        print(f"Predicted label: {CLASS_LABELS.get(prediction, 'Unknown class')}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(passenger)[0]
            print("Estimated probabilities:")
            for class_value, probability in zip(model.classes_, probabilities):
                class_number = int(class_value)
                label = CLASS_LABELS.get(class_number, "Unknown class")
                print(f"  {class_number} ({label}): {probability * 100:.2f}%")
        else:
            print("Class probabilities are not supported by the loaded model.")

        # Pause between simulated arrivals, but never after the final record.
        if record_number < len(stream_records):
            time.sleep(1)

    print("\nStreaming inference simulation completed. Processed 10 records.")


if __name__ == "__main__":
    main()
