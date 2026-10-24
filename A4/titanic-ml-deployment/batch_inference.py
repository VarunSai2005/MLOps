"""Run batch predictions for 20 newly constructed passenger records."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


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
            "Run train_model.py before running batch inference."
        )
    return joblib.load(model_path)


def build_passenger_batch() -> pd.DataFrame:
    """Construct 20 new records for one batch prediction request."""
    # These are illustrative records created for this batch demonstration.
    # They are not selected from train.csv and do not reuse the online example.
    records = [
        {"Pclass": 1, "Sex": "female", "Age": 34.0, "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "C"},
        {"Pclass": 3, "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 8.5, "Embarked": "S"},
        {"Pclass": 2, "Sex": "female", "Age": 41.0, "SibSp": 0, "Parch": 2, "Fare": 32.0, "Embarked": "Q"},
        {"Pclass": 1, "Sex": "male", "Age": 58.0, "SibSp": 1, "Parch": 1, "Fare": 110.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "female", "Age": 19.0, "SibSp": 2, "Parch": 1, "Fare": 18.0, "Embarked": "C"},
        {"Pclass": 2, "Sex": "male", "Age": 31.0, "SibSp": 0, "Parch": 0, "Fare": 24.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male", "Age": None, "SibSp": 0, "Parch": 2, "Fare": 15.5, "Embarked": "S"},
        {"Pclass": 1, "Sex": "female", "Age": 27.0, "SibSp": 1, "Parch": 1, "Fare": 95.0, "Embarked": "Q"},
        {"Pclass": 2, "Sex": "female", "Age": 52.0, "SibSp": 0, "Parch": 0, "Fare": 29.5, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male", "Age": 7.0, "SibSp": 1, "Parch": 2, "Fare": 21.0, "Embarked": "C"},
        {"Pclass": 1, "Sex": "male", "Age": 45.0, "SibSp": 0, "Parch": 0, "Fare": 72.0, "Embarked": "C"},
        {"Pclass": 2, "Sex": "male", "Age": 36.0, "SibSp": 1, "Parch": 0, "Fare": 28.0, "Embarked": None},
        {"Pclass": 3, "Sex": "female", "Age": 24.0, "SibSp": 0, "Parch": 0, "Fare": 7.75, "Embarked": "S"},
        {"Pclass": 2, "Sex": "female", "Age": None, "SibSp": 1, "Parch": 1, "Fare": 35.0, "Embarked": "C"},
        {"Pclass": 3, "Sex": "male", "Age": 63.0, "SibSp": 0, "Parch": 0, "Fare": 9.0, "Embarked": "Q"},
        {"Pclass": 1, "Sex": "female", "Age": 39.0, "SibSp": 0, "Parch": 2, "Fare": 125.0, "Embarked": "S"},
        {"Pclass": 2, "Sex": "male", "Age": 18.0, "SibSp": 2, "Parch": 0, "Fare": 26.5, "Embarked": "C"},
        {"Pclass": 3, "Sex": "female", "Age": 47.0, "SibSp": 1, "Parch": 0, "Fare": 14.0, "Embarked": "S"},
        {"Pclass": 1, "Sex": "male", "Age": 29.0, "SibSp": 0, "Parch": 1, "Fare": 68.0, "Embarked": "Q"},
        {"Pclass": 3, "Sex": "female", "Age": 12.0, "SibSp": 3, "Parch": 1, "Fare": 23.5, "Embarked": None},
    ]
    return pd.DataFrame(records, columns=FEATURES)


def main() -> None:
    """Load the saved pipeline, predict the batch, and save the results."""
    project_root = Path(__file__).resolve().parent
    model = load_model(project_root)
    passengers = build_passenger_batch()

    # Predict all 20 records in one call; the pipeline handles missing values.
    predictions = model.predict(passengers)
    probabilities = None
    if hasattr(model, "predict_proba"):
        # Probability estimates are also obtained for the complete batch at once.
        probabilities = model.predict_proba(passengers)

    results = passengers.copy()
    results.insert(0, "Batch_Record_ID", range(1, len(passengers) + 1))
    results["Predicted_Survived"] = predictions.astype(int)
    results["Predicted_Label"] = [
        CLASS_LABELS.get(int(prediction), "Unknown class")
        for prediction in predictions
    ]

    if probabilities is not None:
        class_to_column = {
            int(class_value): probabilities[:, index]
            for index, class_value in enumerate(model.classes_)
        }
        results["Probability_Did_Not_Survive"] = class_to_column[0]
        results["Probability_Survived"] = class_to_column[1]

    output_path = project_root / "batch_predictions.csv"
    results.to_csv(output_path, index=False)

    print("Batch inference results (model-estimated probabilities):")
    print(results.to_string(index=False))
    print(f"\nSaved batch predictions to: {output_path}")
    print(f"CSV data rows: {len(results)}")


if __name__ == "__main__":
    main()
