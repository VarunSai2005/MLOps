
import pandas as pd
import numpy as np
from datetime import datetime, timezone

RAW_PATH = "data/raw/CSE_Employability_Skill_Gap_Final.csv"
OUT_PATH = "data/processed/skill_gap_features.parquet"

SKILL_GROUPS = {
    "Programming_Languages": [
        "C_Programming", "C++_Programming", "Java", "Python", "JavaScript",
        "SQL", "Rust", "C#",
    ],
    "Web_Dev": [
        "ReactJS", "NodeJS", "REST_API", "Spring_Boot", "ExpressJS",
        "API_Development",
    ],
    "DSA_Core": [
        "Programming_Logic", "Arrays", "Linked_Lists", "Stacks_Queues",
        "Trees", "Graphs", "Dynamic_Programming", "Searching_Sorting",
        "Competitive_Coding",
    ],
    "AI_ML": [
        "Statistics", "Linear_Algebra", "Machine_Learning", "Deep_Learning",
        "NLP", "Computer_Vision", "Data_Preprocessing", "Feature_Engineering",
        "Model_Evaluation", "Data_Visualization", "TensorFlow", "PyTorch",
        "Prompt_Engineering", "LLM_Knowledge",
    ],
    "Cloud_DevOps": [
        "AWS", "Azure", "Google_Cloud", "Docker", "Kubernetes", "MLflow",
        "DVC", "FastAPI", "CI_CD", "Model_Monitoring",
    ],
    "Databases": ["MySQL", "PostgreSQL", "MongoDB", "Firebase", "DBMS"],
    "Tools_Practices": [
        "Git", "GitHub", "Agile", "Jira", "Linux", "VSCode_Productivity",
        "OOP_Concepts", "System_Design",
    ],
    "Core_CS_Security": [
        "Computer_Networks", "Operating_Systems", "Cybersecurity",
        "Network_Security", "Ethical_Hacking", "Linux_Administration",
    ],
    "Soft_Skills": [
        "Communication", "Teamwork", "Leadership", "Problem_Solving",
        "Critical_Thinking", "Time_Management", "Adaptability", "Creativity",
        "Presentation_Skills", "English_Proficiency", "Public_Speaking",
        "Emotional_Intelligence",
    ],
}

PRACTICAL_EXPOSURE_COLS = [
    "Internship_Count", "Projects_Completed", "Open_Source_Contributions",
    "GitHub_Repositories", "Hackathons_Attended", "Certifications",
    "Workshops_Attended",
]

BADGE_MAP = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _min_max_100(series: pd.Series) -> pd.Series:
    """Scale a column to 0-100 using its own min/max.

    The raw skill columns were rated on two different scales (some 1-10,
    some 40-100), so a category average is only meaningful once every
    column is brought onto the same 0-100 footing first.
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["student_id"] = df["Student_ID"]

    all_skill_cols = sorted({c for cols in SKILL_GROUPS.values() for c in cols if c in df.columns})
    normed_skills = df[all_skill_cols].apply(_min_max_100)

    for group, cols in SKILL_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        out[f"{group}_Score"] = normed_skills[present].mean(axis=1).round(2)

    technical_groups = [g for g in SKILL_GROUPS if g != "Soft_Skills"]
    out["Technical_Skill_Score"] = out[[f"{g}_Score" for g in technical_groups]].mean(axis=1).round(2)
    out["Soft_Skill_Score"] = out["Soft_Skills_Score"]

    out["Overall_Skill_Coverage"] = normed_skills.mean(axis=1).round(2)

    out["Skill_Gap_Score"] = (100 - out["Overall_Skill_Coverage"]).round(2)

    practical = df[PRACTICAL_EXPOSURE_COLS].copy()
    normed = (practical - practical.min()) / (practical.max() - practical.min() + 1e-9)
    out["Practical_Exposure_Score"] = (normed.mean(axis=1) * 100).round(2)

    out["HackerRank_Badge_Score"] = df["HackerRank_Badge_Level"].map(BADGE_MAP).fillna(0).astype(int)

    out["Academic_Performance_Score"] = (
        0.4 * df["CGPA"] * 10 +
        0.3 * df["Internal_Marks"] +
        0.3 * df["External_Marks"]
    ).round(2)

    out["CGPA"] = df["CGPA"]
    out["Backlogs"] = df["Backlogs"]
    out["LeetCode_Problems_Solved"] = df["LeetCode_Problems_Solved"]
    out["Resume_Score"] = df["Resume_Score"]
    out["Placement_Readiness_Score"] = df["Placement_Readiness_Score"]
    out["Employability_Index"] = df["Employability_Index"]
    out["Preferred_Domain"] = df["Preferred_Domain"]
    out["Placement_Status"] = df["Placement_Status"]

    out["Skill_Gap_Level"] = df["Skill_Gap_Level"]

    out["event_timestamp"] = pd.Timestamp(datetime.now(timezone.utc)).floor("s")

    out = out.fillna(0)

    return out


def main():
    df_raw = load_raw()
    df_feat = engineer_features(df_raw)
    df_feat.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df_feat)} rows x {df_feat.shape[1]} columns to {OUT_PATH}")
    print(df_feat.head(3).to_string())


if __name__ == "__main__":
    main()
