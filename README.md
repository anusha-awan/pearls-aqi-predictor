# 🌍 Pearls AQI Predictor

## 3-Day Air Quality Index Forecast

Pearls AQI Predictor is an end-to-end machine learning system that predicts Air Quality Index (AQI) for the next 3 days using a serverless data and ML pipeline.

The project uses real-time air-quality data, automated feature engineering, historical backfill, machine-learning model training, Hopsworks Feature Store and Model Registry, GitHub Actions automation, and an interactive Streamlit dashboard.

---

## 🎯 Project Objective

The objective of this project is to build an automated system that:

* Collects air-quality data from an external API
* Generates useful time-based and historical features
* Stores engineered features in a Feature Store
* Uses historical data to train machine-learning models
* Evaluates multiple regression models
* Selects the best-performing model
* Forecasts AQI for the next 72 hours
* Provides model explainability using SHAP
* Generates AQI health alerts
* Displays current and predicted AQI through an interactive dashboard

---

## 🏗️ System Architecture

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

GitHub Actions automates the feature pipeline hourly and the model-training pipeline daily.

---

## 📊 Data & Feature Engineering

The pipeline collects pollutant and air-quality information and generates additional features for machine-learning prediction.

### Input pollutants

* CO
* NO
* NO₂
* O₃
* SO₂
* PM2.5
* PM10
* NH₃

### Engineered features

The project includes:

* Hour
* Day
* Month
* Day of week
* AQI lag features
* PM2.5 lag features
* PM10 lag features
* Rolling AQI statistics
* AQI change rate

A total of **26 model input features** are used for prediction.

---

## 🤖 Machine Learning

Multiple regression models were experimented with:

1. Ridge Regression
2. Random Forest
3. Gradient Boosting

The models are evaluated using:

* MAE
* RMSE
* R²

A chronological 80/20 train-test split is used to respect the time-series nature of the data.

### Best Model

**Random Forest Regressor**

The model is selected using RMSE as the primary selection metric.

---

## 📈 Model Performance

The final Random Forest model achieved approximately:

| Metric |           Score |
| ------ | --------------: |
| MAE    | 3.72 AQI points |
| RMSE   | 6.58 AQI points |
| R²     |          0.9483 |

These metrics represent the model evaluation results used by the dashboard.

---

## 🔮 72-Hour Forecast

The system generates hourly AQI predictions for the next 72 hours.

The dashboard provides:

* Day +1 forecast
* Day +2 forecast
* Day +3 forecast
* Hourly forecast trend
* Daily forecast summary
* Maximum predicted AQI
* AQI health category

---

## 🧠 Model Explainability

SHAP (SHapley Additive exPlanations) is used to explain the model's predictions.

The dashboard provides:

* Global feature importance
* Individual prediction explanation
* Feature contribution direction

This helps identify which variables have the greatest influence on AQI predictions.

---

## 🚨 AQI Health Alerts

The dashboard categorizes predicted AQI levels and displays health-related alerts when AQI reaches higher-risk categories.

This makes the prediction system more useful for end users instead of displaying only numerical predictions.

---

## ⚙️ Automated Pipelines

### Hourly Feature Pipeline

GitHub Actions automatically runs the feature pipeline every hour.

The pipeline:

1. Fetches new AQI data
2. Performs feature engineering
3. Uploads engineered features to Hopsworks

Workflow:

```text
GitHub Actions
      ↓
save_data.py
      ↓
feature_engineering.py
      ↓
upload_features.py
      ↓
Hopsworks Feature Store
```

### Daily Training Pipeline

The training workflow runs automatically every day.

It:

1. Loads historical features
2. Trains and evaluates multiple models
3. Selects the best model
4. Saves the trained model
5. Uploads the model to Hopsworks Model Registry

Workflow:

```text
GitHub Actions
      ↓
train_model.py
      ↓
Model Evaluation
      ↓
Best Model
      ↓
upload_model.py
      ↓
Hopsworks Model Registry
```

---

## 🖥️ Streamlit Dashboard

The interactive dashboard displays:

* Current AQI
* PM2.5
* AQI category
* 72-hour forecast
* Daily forecast summary
* Model performance
* SHAP feature importance
* Individual prediction explanation
* Model information
* Data information
* AQI health alerts

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Gradient Boosting
* Ridge Regression
* SHAP
* Hopsworks Feature Store
* Hopsworks Model Registry
* GitHub Actions
* Streamlit
* Matplotlib
* TensorFlow

---

## 📁 Project Structure

```text
.
├── app.py
├── save_data.py
├── backfill.py
├── feature_engineering.py
├── prepare_training_data.py
├── train_model.py
├── predict_3_days.py
├── evaluate_model.py
├── eda.py
├── shap_analysis.py
├── upload_features.py
├── upload_model.py
├── test_api.py
├── historical_test.py
├── check_data.py
├── aqi_model.pkl
├── model_metadata.pkl
├── model_comparison.csv
├── features.csv
├── historical_aqi.csv
├── training_data.csv
├── predictions_3_days.csv
├── requirements.txt
├── PROJECT_REPORT.md
└── .github/
    └── workflows/
        ├── feature_pipeline.yml
        └── daily_training.yml
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file containing the required API credentials:

```text
OPENWEATHER_API_KEY=your_api_key
HOPSWORKS_API_KEY=your_api_key
```

Do not commit `.env` or API keys to GitHub.

### 6. Run the dashboard

```bash
streamlit run app.py
```

---

## 🔐 Security

API credentials are stored using environment variables and GitHub Actions Secrets.

The `.env` file is excluded from version control through `.gitignore`.

---

## 📌 Project Deliverables

The completed system provides:

* End-to-end AQI prediction system
* Historical feature backfill
* Feature Store integration
* Machine-learning model comparison
* Best-model selection
* Model Registry integration
* 72-hour AQI forecasting
* Automated hourly feature pipeline
* Automated daily training pipeline
* Interactive Streamlit dashboard
* SHAP model explainability
* AQI health alerts
* Project documentation

---

## 👩‍💻 Project

**Pearls AQI Predictor**

Built as part of the 10Pearls Shine Internship project.

---
