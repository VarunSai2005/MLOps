# Curriculum-Industry Skill Feature Store Using Feast

## Student Details

- **Name:** Neeli Setty Venkata Lokesh Babu
- **Register Number:** 231FA04E30
- **Section:** SEC-15

---

## Problem Statement

Engineering curricula and industry hiring requirements evolve at different
speeds, which creates a measurable **skill gap**: students graduate having
been assessed on academics (CGPA, internal/external marks) while employers
evaluate them on a much wider basis — programming languages, DSA, AI/ML,
cloud/DevOps tooling, databases, core CS fundamentals, soft skills, and
practical exposure (internships, projects, open-source work, coding-platform
activity). This project takes a per-student CSE employability dataset that
already carries self/faculty-rated proficiency across 65 individual
technical and soft skills, engineers it into a clean set of curriculum-vs-industry
skill features, and serves those features through a **Feast feature store**
so that the same feature values can be reused consistently for both model
training (historical/offline retrieval) and real-time inference (online
retrieval) — rather than being recomputed ad hoc each time.

---

## Dataset

- **Source file:** `data/raw/CSE_Employability_Skill_Gap_Final.csv`
- **Number of students (rows):** 50,000
- **Number of skills rated:** 65 individual skill columns (1–10 or 40–100
  scale, depending on the skill), spanning 9 categories — Programming
  Languages, Web Dev, DSA/Core, AI/ML, Cloud/DevOps, Databases, Tools &
  Practices, Core CS & Security, and Soft Skills.
- **Dataset columns (121 total):** student identifiers and demographics
  (`Student_ID`, `Age`, `Gender`, `Degree`, `Specialization`); academic
  performance (`CGPA`, `Attendance_Percentage`, `Internal_Marks`,
  `External_Marks`, `Backlogs`, `Class_Rank`, …); the 65 individual skill
  ratings; practical-exposure counters (`Internship_Count`,
  `Projects_Completed`, `Open_Source_Contributions`, `GitHub_Repositories`,
  `LeetCode_Problems_Solved`, `Hackathons_Attended`, `Certifications`, …);
  interview/readiness scores (`Resume_Score`, `Aptitude_Score`,
  `Technical_Interview_Score`, `Placement_Readiness_Score`,
  `Employability_Index`); and outcome fields (`Placement_Status`,
  `Placement_Package_LPA`).
- **Target column:** `Skill_Gap_Level` — a 3-class label (`Low` / `Medium` /
  `High`) describing how large each student's overall skill gap is.
  Class balance: High 37,154 · Medium 12,186 · Low 660 (imbalanced,
  reflecting a real cohort where most students still have a sizeable gap).
- **How the entries were created:** each row represents one CSE student,
  with skill ratings collected from a combination of self-assessment,
  faculty/lab evaluation, and platform-derived metrics (e.g. LeetCode/
  HackerRank activity), alongside official academic and placement records.

---

## Feature Engineering

All engineering lives in [`src/feature_engineering.py`](src/feature_engineering.py)
and outputs `data/processed/skill_gap_features.parquet` — the file Feast
treats as its offline data source.

Because the 65 raw skill columns are rated on two different underlying
scales (most are 1–10, but `DBMS`, `OOP_Concepts`, `System_Design`,
`Prompt_Engineering`, `LLM_Knowledge`, `TensorFlow`, `PyTorch`,
`Public_Speaking`, and `Emotional_Intelligence` are 40–100), every raw skill
column is first **min–max normalised to 0–100 individually**, so that
categories built from a mix of scales aren't skewed by whichever raw column
happened to use the larger range.

| Feast Feature | Meaning |
|---|---|
| `Programming_Languages_Score` | Avg. normalised rating across C, C++, Java, Python, JavaScript, SQL, HTML/CSS |
| `Web_Dev_Score` | Avg. rating across ReactJS, NodeJS, REST API, Spring Boot, ExpressJS, API Development |
| `DSA_Core_Score` | Avg. rating across Arrays, Linked Lists, Stacks/Queues, Trees, Graphs, DP, Searching/Sorting, Competitive Coding |
| `AI_ML_Score` | Avg. rating across Statistics, Linear Algebra, ML, DL, NLP, CV, Data Preprocessing, Feature Engineering, Model Evaluation, Data Viz, TensorFlow, PyTorch, Prompt Engineering, LLM Knowledge |
| `Cloud_DevOps_Score` | Avg. rating across AWS, Azure, GCP, Docker, Kubernetes, MLflow, DVC, FastAPI, CI/CD, Model Monitoring |
| `Databases_Score` | Avg. rating across MySQL, PostgreSQL, MongoDB, Firebase, DBMS |
| `Tools_Practices_Score` | Avg. rating across Git, GitHub, Agile, Jira, Linux, VSCode productivity, OOP, System Design |
| `Core_CS_Security_Score` | Avg. rating across Computer Networks, OS, Cybersecurity, Network Security, Ethical Hacking, Linux Administration |
| `Soft_Skills_Score` | Avg. rating across Communication, Teamwork, Leadership, Problem Solving, Critical Thinking, Time Management, Adaptability, Creativity, Presentation Skills, English Proficiency, Public Speaking, Emotional Intelligence |
| `Technical_Skill_Score` | Mean of all technical category scores above (excludes Soft Skills) |
| `Soft_Skill_Score` | Same value as `Soft_Skills_Score`, kept for a clean technical-vs-soft comparison |
| `Overall_Skill_Coverage` | Mean of all 65 normalised skill columns — how much of the full skill checklist the student covers, 0–100 |
| `Skill_Gap_Score` | `100 − Overall_Skill_Coverage` — see worked example below |
| `Practical_Exposure_Score` | Min-max composite of internships, projects, open-source contributions, GitHub repos, hackathons, certifications, workshops |
| `HackerRank_Badge_Score` | Ordinal encoding of HackerRank badge (Bronze=1 … Platinum=4, none=0) |
| `Academic_Performance_Score` | `0.4×(CGPA×10) + 0.3×Internal_Marks + 0.3×External_Marks` |
| `CGPA`, `Backlogs`, `LeetCode_Problems_Solved`, `Resume_Score`, `Placement_Readiness_Score`, `Employability_Index` | Carried through unchanged from the raw dataset for model context |
| `Preferred_Domain`, `Placement_Status` | Carried through as categorical context |
| `Skill_Gap_Level` | Target label |

### Worked example: how `Skill_Gap_Score` is calculated

For student `CSE000003`:
1. All 65 skill columns are min-max normalised to 0–100.
2. `Overall_Skill_Coverage` = mean of those 65 normalised values = **17.71**.
3. `Skill_Gap_Score = 100 − 17.71 = 82.29`.

A high `Skill_Gap_Score` means the student's demonstrated skills cover only
a small fraction of the full curriculum-vs-industry skill checklist — this
matches the dataset's own label for that student, `Skill_Gap_Level = High`.

### Difference between the original dataset and the feature dataset

The original dataset has **121 raw columns per student**, including 65
individual, granular skill ratings on inconsistent scales, several
interview/readiness sub-scores, and demographic fields not useful for
skill-gap modelling. The engineered feature dataset (`skill_gap_features.parquet`)
condenses this into **27 model-ready columns**: 9 category rollups, 4
composite scores (`Technical_Skill_Score`, `Soft_Skill_Score`,
`Overall_Skill_Coverage`, `Skill_Gap_Score`), 2 activity/badge rollups, an
academic rollup, and a handful of carried-through context columns — all on
a common 0–100 scale, plus the `student_id`/`event_timestamp` keys Feast
requires. This makes the features directly consumable by a model and by
Feast's online store, instead of requiring 65 raw ratings to be
recomputed and re-scaled at inference time.

---

## Feast Architecture

```
Original Dataset (CSV, 50,000 students x 121 columns)
        │
        ▼
Feature Engineering  (src/feature_engineering.py)
        │
        ▼
Parquet Offline Data  (data/processed/skill_gap_features.parquet)
        │
        ▼
Feast FeatureView  (feature_repo/definitions.py → skill_gap_features)
        │
        ├────────────────────────────┐
        ▼                            ▼
Historical Features            Materialization
(get_historical_features)      (feast materialize)
        │                            │
        ▼                            ▼
   Model Training              Online Store (SQLite)
                                     │
                                     ▼
                              Online Retrieval
                           (get_online_features)
                                     │
                                     ▼
                                Prediction
```

---

## Implementation

- **Entity:** `student`, joined on `student_id` (from `Student_ID` in the
  raw CSV) — defined in [`feature_repo/definitions.py`](feature_repo/definitions.py).
- **Data source:** a `FileSource` pointing at
  `data/processed/skill_gap_features.parquet`, with `event_timestamp` as the
  timestamp field. Since the raw dataset is a single snapshot survey (no
  repeated observations per student over time), every row is stamped with
  one ingestion timestamp at feature-engineering time.
- **FeatureView:** `skill_gap_features`, containing the 24 engineered/carried
  columns described above, `ttl=3650 days` (long, since this is a one-off
  cohort snapshot rather than a streaming feature), registered with
  `online=True` so it can be materialized.
- **Registration:** `feast apply` (run from `feature_repo/`) parses
  `definitions.py`, creates the `student` entity and `skill_gap_features`
  FeatureView in the local SQLite registry, and provisions the matching
  online-store table.
- **Historical retrieval:** [`src/historical_retrieval.py`](src/historical_retrieval.py)
  builds an entity dataframe of all 50,000 `student_id` + `event_timestamp`
  pairs and calls `store.get_historical_features(...)`, producing a
  point-in-time-correct training dataset — this is what feeds the model.
- **Materialization:** [`src/materialize.py`](src/materialize.py) runs
  `store.materialize(start_date, end_date)` to push the latest feature
  values for every student from the offline parquet into the SQLite online
  store.
- **Model:** [`src/train_model.py`](src/train_model.py) trains a
  `RandomForestClassifier` (scikit-learn, 200 trees, `class_weight="balanced"`
  to handle the imbalanced target) on the Feast-retrieved historical
  features to predict `Skill_Gap_Level`.
- **Online retrieval:** [`src/online_retrieval.py`](src/online_retrieval.py)
  calls `store.get_online_features(...)` for one or more `student_id`s to
  fetch their latest features from SQLite with low latency.
- **End-to-end inference:** [`src/predict.py`](src/predict.py) chains the
  two together — fetch a student's features from the **online** store, feed
  them into the trained model, and return a predicted `Skill_Gap_Level`
  with class probabilities, simulating how a real placement-readiness
  service would use this feature store.

---

## How to Run

```bash
pip install -r requirements.txt

# 1. Feature engineering (raw CSV -> Feast-ready parquet)
python src/feature_engineering.py

# 2. Register entities/feature views with Feast
cd feature_repo && feast apply && cd ..

# 3. Historical (offline) feature retrieval -> training dataset
python src/historical_retrieval.py

# 4. Train the model
python src/train_model.py

# 5. Materialize features into the online store
python src/materialize.py

# 6. Online feature retrieval
python src/online_retrieval.py

# 7. End-to-end online prediction for a student
python src/predict.py
```

---

## Results

### Historical feature output (sample)

See [`results/historical_features_sample.csv`](results/historical_features_sample.csv)
— a 20-row sample of the point-in-time-correct training data returned by
`get_historical_features()` (full 50,000-row set is used for training via
`data/processed/training_dataset.parquet`).

### Model accuracy

From [`results/model_metrics.json`](results/model_metrics.json), evaluated
on a held-out 20% test split (10,000 students):

| Metric | Value |
|---|---|
| Accuracy | **0.80** |
| Macro F1 | **0.61** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 0.91 | 0.85 | 0.88 | 7,431 |
| Medium | 0.58 | 0.67 | 0.62 | 2,437 |
| Low | 0.23 | 0.55 | 0.33 | 132 |

Top predictive features: `Overall_Skill_Coverage`, `CGPA`,
`Technical_Skill_Score`, `Resume_Score`, `Skill_Gap_Score` — consistent with
the engineered features actually capturing the skill-gap signal (the
`Low` class has weak precision because it makes up only ~1.3% of the
dataset).

### Online feature output (sample)

See [`results/online_features_sample.json`](results/online_features_sample.json)
— features for students `CSE000001`, `CSE000002`, `CSE000003` fetched
directly from Feast's SQLite online store after materialization.

### Final prediction

See [`results/final_prediction.json`](results/final_prediction.json). For
student `CSE000003`, fetched entirely from the **online** store:

```
Skill_Gap_Score: 82.29   Overall_Skill_Coverage: 17.71
Predicted Skill_Gap_Level: High   (P(High) = 0.65, P(Medium) = 0.35)
```

This matches the student's actual recorded `Skill_Gap_Level` of `High`.

---

## Required Analysis

**1. What is the entity in your Feast implementation?**
`student`, keyed by `student_id` (the `Student_ID` field from the raw
dataset, e.g. `CSE000001`).

**2. List the features stored in your FeatureView.**
`Programming_Languages_Score`, `Web_Dev_Score`, `DSA_Core_Score`,
`AI_ML_Score`, `Cloud_DevOps_Score`, `Databases_Score`,
`Tools_Practices_Score`, `Core_CS_Security_Score`, `Soft_Skills_Score`,
`Technical_Skill_Score`, `Soft_Skill_Score`, `Overall_Skill_Coverage`,
`Skill_Gap_Score`, `Practical_Exposure_Score`, `HackerRank_Badge_Score`,
`Academic_Performance_Score`, `CGPA`, `Backlogs`,
`LeetCode_Problems_Solved`, `Resume_Score`, `Placement_Readiness_Score`,
`Employability_Index`, `Preferred_Domain`, `Placement_Status`,
`Skill_Gap_Level` (24 features total, see the Feature Engineering table
above for what each one means).

**3. Explain how one feature was calculated.**
`Skill_Gap_Score` is calculated as `100 − Overall_Skill_Coverage`, where
`Overall_Skill_Coverage` is the mean of all 65 individual skill ratings
after each raw column is independently min-max normalised to a 0–100
scale (needed because some raw ratings are on a 1–10 scale and others on
a 40–100 scale). See the worked example above for student `CSE000003`.

**4. What is the difference between your original dataset and the feature
dataset?**
The original dataset has 121 raw columns including 65 skill ratings on two
different scales plus many auxiliary academic/interview columns. The
feature dataset condenses this to 27 columns: 9 category rollups, 4
composite scores, an academic rollup, activity/badge rollups, and the
Feast keys — all on a consistent 0–100 scale and ready for both training
and low-latency serving. See the full comparison above.

**5. What is the purpose of the offline store?**
The offline store (Feast's `FileSource`, here a parquet file) holds the
full historical feature data used for generating point-in-time-correct
training datasets via `get_historical_features()`. It favours completeness
and correctness over latency — appropriate for batch training.

**6. What is the purpose of the online store?**
The online store (SQLite here) holds only the *latest* feature value per
entity, indexed for fast key-based lookups, so a live inference service can
call `get_online_features()` for a single `student_id` and get a response
in milliseconds — the same feature values, just optimized for serving
rather than for large-scale historical joins.

**7. What is the purpose of `feast apply`?**
`feast apply` reads the Python feature definitions (entities, data sources,
FeatureViews), validates them, and registers them in Feast's registry
(`registry.db`), while also provisioning the corresponding tables in the
online store. It is how Feast becomes "aware" of what features exist and
how to serve them — nothing is retrievable until it's been applied.

**8. What does materialization do?**
`store.materialize(start_date, end_date)` reads feature values from the
offline source (the parquet file) for the given time range and writes the
most recent value per entity into the online store. It is the step that
moves data from "available for batch training" to "available for
low-latency serving."

**9. What is the advantage of retrieving features through Feast instead of
manually calculating them separately during training and prediction?**
Feast guarantees the *same* feature definitions and values are used at both
training and inference time (avoiding training-serving skew), centralises
feature engineering so it isn't duplicated/re-implemented differently in
each pipeline, provides point-in-time-correct joins for historical data
automatically, and gives inference services a fast, simple key-based
lookup API instead of needing to recompute 65 raw skill ratings on the fly.

**10. State two limitations of your current dataset.**
- The skill ratings appear to be largely self-reported/synthetically
  generated rather than derived from independently verified assessments
  (e.g. actual test scores per skill), so they may not fully reflect real
  proficiency.
- The `Skill_Gap_Level` target is heavily imbalanced (74% High, 24%
  Medium, only 1.3% Low), which makes it hard for any model — including
  this one — to learn the `Low` class reliably (see its 0.23 precision).

**11. State two ways your feature store could be improved when more
curriculum and industry evidence becomes available.**
- Add **time-varying** features (e.g. skill ratings per semester instead of
  one snapshot), which would let the offline store's point-in-time joins
  actually matter — right now every row shares one `event_timestamp`, so
  Feast's temporal correctness guarantees are underused.
- Enrich `Preferred_Domain` alignment with **industry job-posting data**
  (e.g. real-time in-demand-skill frequencies scraped from job boards) as
  a new FeatureView, then compute a domain-specific gap score instead of
  one generic `Skill_Gap_Score` per student.

---

## Repository Structure

```
231FA04E30-MLOps-Feast-SkillGap/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/CSE_Employability_Skill_Gap_Final.csv
│   └── processed/
│       ├── skill_gap_features.parquet      # Feast offline data source
│       └── training_dataset.parquet        # full get_historical_features() output
├── feature_repo/
│   ├── feature_store.yaml
│   ├── definitions.py                      # entity, data source, FeatureView
│   └── data/
│       ├── registry.db
│       └── online_store.db
├── src/
│   ├── feature_engineering.py
│   ├── historical_retrieval.py
│   ├── materialize.py
│   ├── online_retrieval.py
│   ├── train_model.py
│   └── predict.py
└── results/
    ├── historical_features_sample.csv
    ├── model_metrics.json
    ├── online_features_sample.json
    ├── final_prediction.json
    ├── skill_gap_model.joblib
    └── label_encoder.joblib
```
