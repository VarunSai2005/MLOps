
import json

from feast import FeatureStore

FEATURE_LIST = [
    "skill_gap_features:Programming_Languages_Score",
    "skill_gap_features:Web_Dev_Score",
    "skill_gap_features:DSA_Core_Score",
    "skill_gap_features:AI_ML_Score",
    "skill_gap_features:Cloud_DevOps_Score",
    "skill_gap_features:Databases_Score",
    "skill_gap_features:Tools_Practices_Score",
    "skill_gap_features:Core_CS_Security_Score",
    "skill_gap_features:Soft_Skills_Score",
    "skill_gap_features:Technical_Skill_Score",
    "skill_gap_features:Soft_Skill_Score",
    "skill_gap_features:Overall_Skill_Coverage",
    "skill_gap_features:Skill_Gap_Score",
    "skill_gap_features:Practical_Exposure_Score",
    "skill_gap_features:HackerRank_Badge_Score",
    "skill_gap_features:Academic_Performance_Score",
    "skill_gap_features:CGPA",
    "skill_gap_features:Backlogs",
    "skill_gap_features:LeetCode_Problems_Solved",
]


def main(student_ids=None):
    store = FeatureStore(repo_path="feature_repo")

    if student_ids is None:
        student_ids = ["CSE000001", "CSE000002", "CSE000003"]

    entity_rows = [{"student_id": sid} for sid in student_ids]

    online_features = store.get_online_features(
        features=FEATURE_LIST,
        entity_rows=entity_rows,
    ).to_dict()

    print(json.dumps(online_features, indent=2, default=str))

    with open("results/online_features_sample.json", "w") as f:
        json.dump(online_features, f, indent=2, default=str)
    print("\nSaved results/online_features_sample.json")


if __name__ == "__main__":
    main()
