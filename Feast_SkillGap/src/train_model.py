
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "Programming_Languages_Score", "Web_Dev_Score", "DSA_Core_Score",
    "AI_ML_Score", "Cloud_DevOps_Score", "Databases_Score",
    "Tools_Practices_Score", "Core_CS_Security_Score", "Soft_Skills_Score",
    "Technical_Skill_Score", "Soft_Skill_Score", "Overall_Skill_Coverage",
    "Skill_Gap_Score", "Practical_Exposure_Score", "HackerRank_Badge_Score",
    "Academic_Performance_Score", "CGPA", "Backlogs",
    "LeetCode_Problems_Solved", "Resume_Score", "Placement_Readiness_Score",
    "Employability_Index",
]
TARGET_COL = "Skill_Gap_Level"


def main():
    df = pd.read_parquet("data/processed/training_dataset.parquet")

    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop feature importances:")
    for name, imp in importances[:8]:
        print(f"  {name:35s} {imp:.4f}")

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "top_features": [{"feature": n, "importance": float(i)} for n, i in importances[:8]],
        "label_classes": le.classes_.tolist(),
    }
    with open("results/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, "results/skill_gap_model.joblib")
    joblib.dump(le, "results/label_encoder.joblib")
    print("\nSaved results/model_metrics.json, results/skill_gap_model.joblib, results/label_encoder.joblib")


if __name__ == "__main__":
    main()
