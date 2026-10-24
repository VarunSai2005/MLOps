
# ML Model Deployment Strategies and Model Serving using FastAPI

This project is a starter structure for a college experiment using the official
Kaggle Titanic dataset. The experiment will train a classification model and
compare several ways to make that model available for predictions.

## Official dataset

Download the dataset from the official Kaggle competition page:

<https://www.kaggle.com/competitions/titanic/data>

Place the required, authentic dataset files in `data/`. Do not commit credentials,
fabricated records, fake models, dummy predictions, or unverified accuracy
results.

## Experiment tasks

1. **Dataset preparation** — obtain the official data, inspect its schema, clean
   missing values, and select reproducible features.
2. **Model training and evaluation** — train a scikit-learn classifier, evaluate
   it using a documented procedure, and persist the real trained artifact.
3. **Online inference** — implement a single-record prediction workflow for
   low-latency requests.
4. **Batch inference** — implement prediction for a CSV or other finite batch and
   save the real outputs for analysis.
5. **Streaming inference** — process passenger records incrementally and document
   the assumptions and trade-offs of the streaming approach.
6. **FastAPI model serving** — expose health and prediction endpoints, then use
   the client to exercise the running service and record observations.

The completed experiment should compare the operational trade-offs of the
inference strategies, including latency, throughput, reproducibility, and
deployment complexity. Results must be generated from the official data and
actual executions rather than filled in ahead of time.

## Project structure

```text
titanic-ml-deployment/
├── data/
│   └── .gitkeep
├── screenshots/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── train_model.py
├── online_inference.py
├── batch_inference.py
├── streaming_inference.py
├── app.py
├── client.py
├── observations.md
├── requirements.txt
├── .gitignore
└── README.md
```

The Python files intentionally contain TODOs only. The ML workflow, model
artifact, predictions, and measured results will be added in later tasks.

## Environment setup

From PowerShell, enter this project directory and create a virtual environment:

```powershell
cd .\titanic-ml-deployment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

If PowerShell blocks script activation for the current user, run this once in
PowerShell and then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
