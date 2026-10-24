"""Run online inference for one newly constructed passenger record."""

from pathlib import Path

import joblib
import pandas as pd


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CLASS_LABELS = {
    0: "Did not survive",
    1: "Survived",
}


def load_model(project_root: Path):
    """Load the already-trained pipeline used for online predictions."""
    model_path = project_root / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run train_model.py before running online inference."
        )
    return joblib.load(model_path)


def main() -> None:
    """Create one unseen record and obtain a prediction from the saved pipeline."""
    project_root = Path(__file__).resolve().parent
    model = load_model(project_root)

    # This is a newly constructed demonstration record, not a row from train.csv.
    passenger = pd.DataFrame(
        [
            {
                "Pclass": 2,
                "Sex": "female",
                "Age": 29.0,
                "SibSp": 0,
                "Parch": 1,
                "Fare": 26.0,
                "Embarked": "C",
            }
        ],
        columns=FEATURES,
    )

    # The saved pipeline performs the same preprocessing used during training.
    prediction = int(model.predict(passenger)[0])
    print("Input passenger details (new demonstration record):")
    print(passenger.to_string(index=False))
    print(f"\nPredicted class: {prediction}")
    print(f"Predicted class label: {CLASS_LABELS.get(prediction, 'Unknown class')}")

    # LogisticRegression supports predict_proba; identify these as estimates.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(passenger)[0]
        print("\nEstimated probability for each class:")
        for class_value, probability in zip(model.classes_, probabilities):
            class_number = int(class_value)
            class_label = CLASS_LABELS.get(class_number, "Unknown class")
            print(
                f"  {class_number} ({class_label}): "
                f"{probability * 100:.2f}%"
            )
    else:
        print("\nClass probabilities are not supported by the loaded model.")


if __name__ == "__main__":
    main()
