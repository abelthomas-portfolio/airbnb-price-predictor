# Airbnb Price Prediction

This repository contains a comprehensive data science project focused on predicting Airbnb listing prices in San Francisco using supervised machine learning techniques. The goal is to develop a reliable, interpretable model that provides accurate price estimations based on listing features. The project has been fully documented, evaluated, and deployed as a Streamlit web application.

---

## 📌 Project Objective

To develop and deploy a machine learning model that predicts the price of Airbnb listings in San Francisco based on key features such as room type, number of accommodations, number of bedrooms, review scores, and more. The predictions aim to support hosts in pricing decisions and offer insights to prospective guests.

---

## 🗂️ Repository Structure

```
airbnb-price-prediction/
│
├── app/                            # Streamlit app for deployment
│   └── app.py
│
├── notebooks/                      # Jupyter notebooks for analysis and modeling
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_model_selection.ipynb
│   └── 04_model_testing.ipynb
│
├── models/                         # Serialized final model
│   └── final_model.pkl
│
├── data/
│   ├── raw/
│   │   └── converted.csv
│   └── processed/
│       ├── preprocessed_airbnb_data.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── reports/
│   └── final_report.pdf            # Full documentation of methodology and results
│
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔍 Methodology Overview

### 1. Data Collection & Preprocessing
- Source: Airbnb public listings data for San Francisco
- Handled missing values, outliers (capped at 99th percentile), and performed log transformation on the price variable
- One-hot encoding for categorical variables; binary features encoded as 0/1

### 2. Feature Engineering & Selection
- Feature correlations and multicollinearity analysis guided the selection process
- Feature importance assessed using Random Forest
- Final features: `bedrooms`, `accommodates`, `review_scores_rating`, `room_type_Private room`, `room_type_Shared room`, and `bathrooms`

### 3. Model Development
- Models evaluated: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting
- Evaluation metric: Mean Squared Error (MSE)
- Final model: Gradient Boosting Regressor selected after hyperparameter tuning using GridSearchCV

### 4. Model Testing & Interpretation
- Performance tested on realistic and edge case listings
- SHAP used to interpret and explain predictions
- Final MSE: **0.1897**, R²: **0.5952**

---

## 🚀 Application Deployment

The final model has been deployed using **Streamlit**, offering users a simple interface to input listing details and receive an estimated price.

To run the app locally:

```bash
streamlit run app/app.py
```
Or check out the deployed version [here](#) *(google.com)*

---

## 📈 Key Technologies

- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, SHAP, cloudpickle, Streamlit
- **Tools:** Jupyter, Git, Streamlit Cloud

---

## 📘 Documentation

The complete methodology, rationale, analysis steps, and evaluation results are provided in:
- [`reports/"Predicting Airbnb Prices - A Machine Learning Approach.pdf"`](reports/"Predicting Airbnb Prices - A Machine Learning Approach.pdf")

---

## 👤 Author

**Abel Thomas**  
GitHub: [@abelthomas-portfolio](https://github.com/abelthomas-portfolio)

