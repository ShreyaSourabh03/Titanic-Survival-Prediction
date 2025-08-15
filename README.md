#  🚢 Titanic Survival Prediction — End‑to‑End ML Project

A polished, hands-on machine learning pipeline that predicts passenger survival on the Titanic using the classic Kaggle dataset. Designed for clarity, reproducibility, and fast iteration—with clean preprocessing, focused feature engineering, and model evaluation.

# ✨ Highlights

1.) End‑to‑end workflow: load → preprocess → engineer → train → evaluate → predict

2.) Clean handling of missing values and categorical encoding without leakage

3.)Baseline and improved models with accuracy and F1 tracking

4.)Reproducible notebook structure and exportable predictions

# 🧰 Tech Stack

Language: Python 3.x

Libraries: NumPy, pandas, scikit‑learn, seaborn, matplotlib


# 📦 Dataset

Source: Kaggle Titanic (train.csv, test.csv)

Target: Survived (0/1)

Core features used:

   Numerical: Pclass, Age, SibSp, Parch, Fare
   
   Categorical: Sex, Embarked
   
   Typically dropped/noisy: Name, Ticket, Cabin (can derive features)


# 📁 Project Structure


titanic-ml/
├─ Titanic-Project.ipynb        # Main notebook: EDA → preprocessing → modeling → outputs
├─ data/
│  ├─ train.csv
│  └─ test.csv
├─ outputs/
│  └─ submission.csv            # Generated predictions (optional)
├─ requirements.txt             # Dependencies (optional)
└─ README.md                    # This file

# 🔎 Methodology

-> Data Loading & Sanity Checks

-> Inspect shapes, dtypes, and missingness patterns.

->Validate target distribution and key feature ranges.

->Missing Value Strategy

->Age → median (or model-based) imputation

->Embarked → most frequent category

->Fare (test) → median if missing

->Cabin → excluded in baseline; optional HasCabin flag

->Feature Engineering

   Sex → binary encoding (male=1, female=0)
    
   Embarked → one‑hot (Q, S; with appropriate reference handling)
   
   Numerical features retained: Pclass, Age, SibSp, Parch, Fare


# Modeling

Baseline: Logistic Regression or RandomForestClassifier

Metrics: Accuracy (primary), F1 (class‑imbalance aware)

Selection: Choose model by CV performance and validation stability

Inference & Submission

Retrain best model on full training data

Predict on test set

Save submission.csv with PassengerId and Survived

# 📊 Metrics & Expectations

Primary: Accuracy

Secondary: F1‑score (macro/weighted)

Typical outcomes:

Logistic Regression → strong, interpretable baseline

Tree‑based models → higher potential with tuned hyperparams

Note: Exact scores vary by features, seeds, and preprocessing choices; rely on CV for stability.
Exported predictions align PassengerId correctly with Survived

# 👩💻 Author
Shreya Sourabh

Interests: Machine Learning, Data Science, applied modeling, deployment

Collaboration welcome—issues and PRs encouraged
