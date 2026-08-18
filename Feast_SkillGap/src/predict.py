
import json

import joblib
import pandas as pd
from feast import FeatureStore

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


def predict_for_student(student_id: str) -> dict:
    store = FeatureStore(repo_path="feature_repo")
    model = joblib.load("results/skill_gap_model.joblib")
    le = joblib.load("results/label_encoder.joblib")

    features = [f"skill_gap_features:{c}" for c in FEATURE_COLS]
    online = store.get_online_features(
        features=features, entity_rows=[{"student_id": student_id}]
    ).to_dict()

    row = {c: online[c][0] for c in FEATURE_COLS}
    X = pd.DataFrame([row])[FEATURE_COLS]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]

    result = {
        "student_id": student_id,
        "online_features": row,
        "predicted_skill_gap_level": label,
        "class_probabilities": dict(zip(le.classes_, proba.tolist())),
    }
    return result


def main():
    result = predict_for_student("CSE000003")
    print(json.dumps(result, indent=2, default=str))
    with open("results/final_prediction.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved results/final_prediction.json")


if __name__ == "__main__":
    main()
