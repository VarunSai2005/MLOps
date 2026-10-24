"""Train and persist the Titanic classification model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Survived"
REQUIRED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
}
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def find_dataset(project_root: Path) -> Path:
    """Find the existing official dataset without copying or overwriting it."""
    candidates = (
        project_root / "data" / "train.csv",
        project_root.parent / "train.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched_paths = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Titanic train.csv was not found. Searched:\n" + searched_paths
    )


def build_pipeline() -> Pipeline:
    """Build the preprocessing and classifier pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    """Load data, train and evaluate the model, and save the fitted pipeline."""
    project_root = Path(__file__).resolve().parent
    dataset_path = find_dataset(project_root)
    print(f"Loading dataset from: {dataset_path}")

    data = pd.read_csv(dataset_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required Titanic columns: "
            + ", ".join(missing_columns)
        )
    if data[TARGET].isna().any():
        raise ValueError("The target column 'Survived' contains missing values.")

    selected_data = data[FEATURES + [TARGET]]
    print(f"Dataset shape: {data.shape}")
    print("Missing values in selected features:")
    print(selected_data[FEATURES].isna().sum().to_string())
    print("Target class distribution:")
    print(data[TARGET].value_counts().sort_index().to_string())

    X = selected_data[FEATURES]
    y = selected_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    model_path = project_root / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved fitted pipeline to: {model_path}")


if __name__ == "__main__":
    main()
