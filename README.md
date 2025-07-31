# Airbnb Listing Analysis: Logistic Regression Model

**Author:** Angel Espinosa

---

📌 **Project Overview**  
This project is part of the ML Life Cycle Lab 5 assignment, focused on evaluating and deploying a machine learning model. It demonstrates the full pipeline from data loading to model evaluation and persistence using logistic regression to predict Airbnb superhosts.

---

🧠 **Problem Statement**  
Predict whether an Airbnb host is a **superhost** based on listing features.

- **Type of Problem:** Supervised binary classification  
- **Target Variable:** `host_is_superhost` (True/False)

---

📊 **Dataset**  
- File: `airbnbData_train.csv` (in `data_LR` folder)  
- Dataset is preprocessed: one-hot encoded categoricals, scaled numericals, missing values imputed  
- Features include host info, listing counts, neighborhood, room types, and more

---

🔍 **Exploratory Data Analysis (EDA) & Feature Selection**  
- Dataset overview and feature identification  
- Selected top 5 features using `SelectKBest`:  
  `host_response_rate`, `number_of_reviews`, `number_of_reviews_ltm`, `number_of_reviews_l30d`, `review_scores_cleanliness`

---

⚙️ **Modeling Approach**  
- Logistic Regression with default hyperparameter (`C=1.0`)  
- Hyperparameter tuning with GridSearchCV across 10 values of `C`  
- Retrained logistic regression with optimal `C=100`  
- Compared models via precision-recall and ROC curves  
- Evaluated models using accuracy, confusion matrices, and AUC

---

🔁 **Model Tuning & Evaluation**  
- Used 5-fold cross-validation during hyperparameter search  
- Best model achieved ROC AUC of ~0.8235, slightly improving over default (~0.8229)  
- Feature-selected model yielded an AUC of ~0.797

---

📈 **Evaluation Metrics Summary**  
| Metric          | Default Model | Optimized Model (C=100) | Feature-Selected Model |
|-----------------|---------------|------------------------|-----------------------|
| Accuracy        | (See notebook)| (See notebook)          | (See notebook)         |
| ROC AUC         | 0.8229        | 0.8235                 | 0.797                  |

---

✅ **Model Persistence**  
- Saved best logistic regression model as `logistic_model_best.pkl` using `pickle`  
- Demonstrated loading and using the saved model for prediction

---

📂 **Repository Contents**  
- `ModelSelectionForLogisticRegression.ipynb` — Jupyter notebook with full analysis  
- `airbnbData_train.csv` — Dataset  
- `logistic_model_best.pkl` — Saved logistic regression model  
- `README.md` — This file

---

🙌 **Acknowledgments**  
Developed as part of the ML Life Cycle Lab 5 assignment in the ML Foundations curriculum.

---

## Installation & Usage

1. Clone the repo:  
   ```bash
   git clone https://github.com/aespinosa221120/airbnb-listing-analysis.git
   cd airbnb-listing-analysis
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook ModelSelectionForLogisticRegression.ipynb


  
   
