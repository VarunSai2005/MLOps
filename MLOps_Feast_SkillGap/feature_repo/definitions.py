
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

student = Entity(
    name="student",
    join_keys=["student_id"],
    description="A single CSE student in the curriculum-industry skill-gap study",
)

skill_gap_source = FileSource(
    name="skill_gap_source",
    path="../data/processed/skill_gap_features.parquet",
    timestamp_field="event_timestamp",
)

skill_gap_features_view = FeatureView(
    name="skill_gap_features",
    entities=[student],
    ttl=timedelta(days=3650),  
    schema=[
        Field(name="Programming_Languages_Score", dtype=Float32),
        Field(name="Web_Dev_Score", dtype=Float32),
        Field(name="DSA_Core_Score", dtype=Float32),
        Field(name="AI_ML_Score", dtype=Float32),
        Field(name="Cloud_DevOps_Score", dtype=Float32),
        Field(name="Databases_Score", dtype=Float32),
        Field(name="Tools_Practices_Score", dtype=Float32),
        Field(name="Core_CS_Security_Score", dtype=Float32),
        Field(name="Soft_Skills_Score", dtype=Float32),
        Field(name="Technical_Skill_Score", dtype=Float32),
        Field(name="Soft_Skill_Score", dtype=Float32),
        Field(name="Overall_Skill_Coverage", dtype=Float32),
        Field(name="Skill_Gap_Score", dtype=Float32),
        Field(name="Practical_Exposure_Score", dtype=Float32),
        Field(name="HackerRank_Badge_Score", dtype=Int64),
        Field(name="Academic_Performance_Score", dtype=Float32),
        Field(name="CGPA", dtype=Float32),
        Field(name="Backlogs", dtype=Int64),
        Field(name="LeetCode_Problems_Solved", dtype=Int64),
        Field(name="Resume_Score", dtype=Int64),
        Field(name="Placement_Readiness_Score", dtype=Int64),
        Field(name="Employability_Index", dtype=Int64),
        Field(name="Preferred_Domain", dtype=String),
        Field(name="Placement_Status", dtype=String),
        Field(name="Skill_Gap_Level", dtype=String),
    ],
    online=True,
    source=skill_gap_source,
    description="Engineered curriculum-industry skill-gap features per student",
)
