# Pearls AQI Predictor — Final Project Report

## 1. Project Overview

Pearls AQI Predictor is an end-to-end machine learning system designed to forecast Air Quality Index (AQI) for the next 3 days.

The system collects air-quality data from an external API, performs automated feature engineering, stores engineered features in Hopsworks Feature Store, trains and evaluates machine-learning models, registers the trained model in Hopsworks Model Registry, generates 72-hour AQI forecasts, provides SHAP-based model explainability, and presents the results through an interactive Streamlit dashboard.

The project was developed as part of the 10Pearls Shine Internship.

---

## 2. Project Objective

The main objective was to develop a scalable and automated AQI forecasting system using a serverless-oriented architecture.

The system was designed to:

* Collect real-time air-quality data.
* Generate time-based and historical features.
* Create training data through historical backfill.
* Store engineered features in a Feature Store.
* Experiment with multiple machine-learning models.
* Evaluate models using MAE, RMSE, and R².
* Select the best-performing model.
* Store the trained model in a Model Registry.
* Generate AQI forecasts for the next 72 hours.
* Explain model predictions using SHAP.
* Provide AQI health alerts.
* Automate feature collection and model training.
* Display results through an interactive dashboard.

---

## 3. System Architecture

The overall system follows this pipeline:

```text
External AQI API
       ↓
Data Collection
       ↓
Feature Engineering
       ↓
Hopsworks Feature Store
       ↓
Historical Backfill
       ↓
Training Dataset
       ↓
Model Training & Evaluation
       ↓
Best Model Selection
       ↓
Hopsworks Model Registry
       ↓
72-Hour AQI Forecast
       ↓
Streamlit Dashboard
```

GitHub Actions is used to automate recurring pipeline execution.

---

## 4. Data Collection

The project uses an external air-quality API to collect pollutant information.

The collected pollutant variables include:

* CO
* NO
* NO₂
* O₃
* SO₂
* PM2.5
* PM10
* NH₃

The data collection process is implemented in Python and stores the collected information for further processing.

The pipeline is designed to run automatically through GitHub Actions.

---

## 5. Historical Backfill

Historical data was generated to provide sufficient observations for model training.

The backfill process collected historical air-quality observations and produced a training dataset containing thousands of records.

The resulting datasets include:

* `historical_aqi.csv`
* `aqi_data.csv`
* `training_data.csv`
* `features.csv`

This historical data provides the basis for supervised machine-learning training.

---

## 6. Feature Engineering

Feature engineering was performed to transform raw air-quality data into useful machine-learning inputs.

### Time-based features

The system includes:

* Hour
* Day
* Month
* Day of week

### Historical features

Lag-based features were created to capture previous AQI and pollutant values, including:

* AQI lag 1
* AQI lag 3
* AQI lag 6
* AQI lag 12
* AQI lag 24
* AQI lag 48
* AQI lag 72
* PM2.5 lag 1
* PM10 lag 1

### Rolling features

Rolling AQI statistics were also created:

* AQI rolling 6
* AQI rolling 24
* AQI rolling 72

An AQI change feature was also included to represent recent changes in air quality.

A total of **26 features** are used as model inputs.

---

## 7. Feature Store

Hopsworks Feature Store was integrated into the project to provide centralized storage for engineered features.

The feature pipeline uploads processed features to the Feature Store.

The local feature dataset is also maintained as a fallback when the Hopsworks query service cannot be reached.

This makes the local development and prediction workflow more robust against temporary Feature Store connectivity problems.

---

## 8. Exploratory Data Analysis

Exploratory Data Analysis was performed using Python, Pandas, and Matplotlib.

The EDA process examined:

* Dataset shape
* Basic statistical information
* AQI distribution
* AQI variation over time
* Average AQI by hour
* Average AQI by day of week

The final EDA dataset contained **8,378 processed observations across 29 columns**.

The AQI distribution was also examined to understand the distribution of air-quality categories in the dataset.

EDA visualizations were generated and saved as image files for documentation and analysis.

---

## 9. Machine Learning Models

Multiple regression algorithms were experimented with:

### Ridge Regression

A Ridge Regression model with feature scaling was implemented as a baseline linear model.

### Random Forest

A Random Forest Regressor was trained using multiple decision trees and was evaluated as the primary nonlinear model.

### Gradient Boosting

A Gradient Boosting Regressor was also evaluated to provide another nonlinear modelling approach.

The models were trained using a chronological 80/20 train-test split so that future observations were not used to train the model before evaluation.

---

## 10. Model Evaluation

Model performance was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² score

The model-training pipeline compares the candidate models and selects the model with the lowest RMSE.

The resulting comparison is stored in:

```text
model_comparison.csv
```

The selected trained model is saved as:

```text
aqi_model.pkl
```

---

## 11. Best Model

The Random Forest Regressor was selected as the best-performing model in the final training pipeline.

The selection is based on RMSE.

The trained model was subsequently uploaded to the Hopsworks Model Registry.

---

## 12. Model Registry

Hopsworks Model Registry was integrated into the project for model versioning and storage.

The trained Random Forest model was successfully uploaded using:

```text
upload_model.py
```

The registered model is named:

```text
aqi_random_forest
```

A model version was successfully created in the Hopsworks Model Registry.

---

## 13. 72-Hour AQI Forecasting

The system generates hourly predictions for the next 72 hours.

The forecasting component provides:

* Day +1 prediction
* Day +2 prediction
* Day +3 prediction
* Hourly AQI forecast
* Daily forecast summaries
* Maximum predicted AQI

The forecast is displayed through the Streamlit dashboard.

---

## 14. SHAP Explainability

SHAP (SHapley Additive exPlanations) was implemented to make the machine-learning model more interpretable.

Two forms of explanation are provided:

### Global Feature Impact

The system calculates average absolute SHAP values to identify which features have the greatest overall influence on predictions.

### Individual Prediction Explanation

The dashboard also provides an explanation for an individual prediction, showing which features push the prediction higher or lower.

This improves transparency and helps users understand the factors influencing the model.

---

## 15. AQI Health Alerts

The dashboard categorizes AQI predictions into health-related categories.

Alerts are displayed when predicted AQI reaches levels that may present increased health concerns.

This allows the application to provide actionable information rather than displaying only numerical AQI predictions.

---

## 16. Automated Feature Pipeline

GitHub Actions is used to automate the feature pipeline.

The workflow is defined in:

```text
.github/workflows/feature_pipeline.yml
```

The pipeline is scheduled to run every hour.

The workflow performs:

```text
save_data.py
      ↓
feature_engineering.py
      ↓
upload_features.py
```

This enables the system to continuously collect and process new air-quality data.

---

## 17. Automated Training Pipeline

A separate GitHub Actions workflow automates model training.

The workflow is defined in:

```text
.github/workflows/daily_training.yml
```

The training workflow performs:

```text
train_model.py
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
upload_model.py
      ↓
Hopsworks Model Registry
```

The workflow is scheduled to run daily and can also be triggered manually.

---

## 18. Streamlit Dashboard

An interactive Streamlit dashboard was developed as the user-facing component of the system.

The dashboard displays:

* Current AQI
* AQI category
* PM2.5
* Latest data timestamp
* 72-hour AQI forecast
* Day +1, Day +2 and Day +3 forecast cards
* Daily forecast summary
* Model performance
* SHAP global feature impact
* Individual prediction explanation
* Model information
* Data information
* AQI health alerts

The dashboard provides a single interface for viewing both current air quality and future predictions.

---

## 19. Technologies Used

The project uses the following technologies:

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Ridge Regression
* Gradient Boosting
* SHAP
* Matplotlib
* TensorFlow
* Hopsworks Feature Store
* Hopsworks Model Registry
* GitHub Actions
* Streamlit
* Joblib
* Python-dotenv

---

## 20. Project Files

Important project components include:

```text
app.py
save_data.py
backfill.py
feature_engineering.py
prepare_training_data.py
train_model.py
predict_3_days.py
evaluate_model.py
eda.py
shap_analysis.py
upload_features.py
upload_model.py
requirements.txt
model_comparison.csv
model_metadata.pkl
aqi_model.pkl
features.csv
historical_aqi.csv
training_data.csv
predictions_3_days.csv
PROJECT_REPORT.md
README.md
```

Automation workflows are located in:

```text
.github/workflows/
```

---

## 21. Security

API credentials are stored using environment variables rather than being directly embedded in source code.

The `.env` file is excluded from version control through `.gitignore`.

GitHub Actions uses repository secrets for required API credentials.

Sensitive credentials should not be included in the public repository.

---

## 22. Challenges and Solutions

### Hopsworks Query Service Timeout

During local model training, the Hopsworks Feature Store Query Service occasionally returned a timeout error.

To maintain functionality, the training pipeline was designed to fall back to the locally available engineered dataset when the Feature Store could not be read.

This allowed model training to continue while preserving Hopsworks integration in the overall architecture.

### Time-Series Data

A chronological train-test split was used instead of random splitting to avoid mixing future observations into the training data.

### Model Selection

Multiple models were evaluated rather than relying on a single algorithm. The model with the lowest RMSE was selected automatically.

---

## 23. Limitations

The forecasting system is based on the available historical and real-time air-quality data.

Prediction quality can be affected by:

* Availability and quality of external API data
* Changes in local environmental conditions
* Sudden pollution events
* Limited historical coverage
* Temporary cloud service connectivity issues

The predictions should therefore be considered forecasts rather than guaranteed future AQI values.

---

## 24. Final Deliverables

The completed project provides the required major components:

1. End-to-end AQI prediction system
2. Historical data backfill
3. Automated feature engineering
4. Feature Store integration
5. Multiple machine-learning models
6. Model evaluation using MAE, RMSE and R²
7. Best-model selection
8. Model Registry integration
9. 72-hour AQI forecasting
10. SHAP model explainability
11. AQI health alerts
12. Automated hourly feature pipeline
13. Automated daily training pipeline
14. Interactive Streamlit dashboard
15. Project documentation

---

## 25. Conclusion

Pearls AQI Predictor provides an end-to-end machine-learning workflow for forecasting air quality over a 72-hour horizon.

The project combines automated data collection, feature engineering, historical backfill, Feature Store integration, model experimentation, model evaluation, Model Registry versioning, automated pipelines, explainable AI, health alerts, and an interactive Streamlit dashboard.

The resulting system demonstrates the complete workflow from raw air-quality data to an automated and user-facing AQI forecasting application.
