
import pandas as pd
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
    "skill_gap_features:Resume_Score",
    "skill_gap_features:Placement_Readiness_Score",
    "skill_gap_features:Employability_Index",
    "skill_gap_features:Preferred_Domain",
    "skill_gap_features:Placement_Status",
    "skill_gap_features:Skill_Gap_Level",
]


def get_entity_df() -> pd.DataFrame:
    processed = pd.read_parquet("data/processed/skill_gap_features.parquet")
    return processed[["student_id", "event_timestamp"]]


def main():
    store = FeatureStore(repo_path="feature_repo")
    entity_df = get_entity_df()

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_LIST,
    ).to_df()

    print(f"Retrieved historical features for {len(training_df)} students")
    print(training_df.head(5).to_string())

    training_df.head(20).to_csv("results/historical_features_sample.csv", index=False)
    training_df.to_parquet("data/processed/training_dataset.parquet", index=False)
    print("Saved results/historical_features_sample.csv (20-row sample) and data/processed/training_dataset.parquet (full)")


if __name__ == "__main__":
    main()
