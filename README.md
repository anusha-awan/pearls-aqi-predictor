# 🌍 Pearls AQI Predictor

## 3-Day Air Quality Index Forecast

Pearls AQI Predictor is an end-to-end machine learning system that predicts the Air Quality Index (AQI) for the next 3 days using air-quality data and machine learning.

The project uses real-time air-quality data, historical data collection, feature engineering, machine-learning model training, model evaluation, Hopsworks Feature Store and Model Registry exploration, GitHub Actions automation, SHAP explainability, and an interactive Streamlit dashboard.

The final production prediction pipeline uses a validated local dataset and a **Gradient Boosting Regressor**. Hopsworks was successfully connected and its Feature Store was created, but the final Feature Store data upload encountered a storage-layer error, so the validated local dataset was used for final training and prediction.

---

## 🎯 Project Objective

The objective of this project is to build an automated system that:

* Collects air-quality data from an external API
* Generates useful time-based and historical features
* Prepares historical data for machine learning
* Experiments with Hopsworks Feature Store
* Uses historical data to train machine-learning models
* Evaluates multiple regression models
* Selects the best-performing model
* Forecasts AQI for the next 72 hours
* Provides model explainability using SHAP
* Generates AQI health alerts
* Displays current and predicted AQI through an interactive dashboard
* Automates project tasks using GitHub Actions
* Deploys the dashboard online

---

## 🏗️ System Architecture

The final working application flow is:

```text
OpenWeather API
       ↓
Live Air Quality Data
       ↓
Python Data Processing
       ↓
Feature Engineering
       ↓
AQI Calculation
       ↓
Gradient Boosting Model
       ↓
72-Hour Recursive Forecast
       ↓
SHAP Explainability
       ↓
Streamlit Dashboard
       ↓
Online Deployment
```

Hopsworks was also explored for Feature Store and Model Registry functionality.

GitHub Actions was used for project automation.

---

## 📊 Data & Feature Engineering

The project collects pollutant and air-quality information and generates additional features for machine-learning prediction.

### Input Pollutants

* CO
* NO
* NO₂
* O₃
* SO₂
* PM2.5
* PM10
* NH₃

### Engineered Features

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

The final production model uses **26 model input features**.

The validated local dataset contained **8,401 valid rows and 28 total columns**, with 26 columns/features used as production model inputs.

---

## 🤖 Machine Learning

Multiple regression models were trained and compared:

1. Ridge Regression
2. Random Forest
3. Gradient Boosting

The models were evaluated using:

* MAE
* RMSE
* R²

A chronological 80/20 train-test split was used to respect the time-series nature of the data.

### Best Model

**Gradient Boosting Regressor**

Gradient Boosting achieved the best overall evaluation results among the tested models and was selected as the final production model.

---

## 📈 Model Performance

The model evaluation results were:

| Model                 |        MAE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| Random Forest         |     3.6747 |     6.8945 |     0.9433 |
| **Gradient Boosting** | **3.5714** | **6.0362** | **0.9565** |
| Ridge Regression      |     4.9105 |     7.3455 |     0.9356 |

Gradient Boosting achieved:

* Lowest MAE
* Lowest RMSE
* Highest R²

Therefore, **Gradient Boosting Regressor** was selected as the final production model.

---

## 🔮 72-Hour Forecast

The system generates hourly AQI predictions for the next 72 hours.

The prediction process uses recursive forecasting, where predicted future values can become part of the input for later predictions.

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
* Feature contribution information

This helps identify which variables have greater influence on AQI predictions.

---

## 🚨 AQI Health Alerts

The dashboard categorizes AQI levels and displays health-related alerts when AQI reaches higher-risk categories.

This makes the system more useful for end users instead of displaying only numerical predictions.

---

## ⚙️ Automation with GitHub Actions

GitHub Actions was used to automate project tasks.

The project includes workflow files for feature processing and model-training operations.

### Feature Pipeline

The feature pipeline is designed to:

1. Fetch AQI data
2. Perform feature engineering
3. Process the resulting data
4. Attempt cloud feature storage operations

### Daily Training Pipeline

The training workflow is designed to:

1. Load historical features
2. Train and evaluate multiple models
3. Select the best-performing model
4. Save the trained model
5. Support model management operations

### Important Hopsworks Limitation

Hopsworks Feature Store was successfully connected and the `aqi_features_v2` Feature Store was created.

However, the final data upload encountered the following storage-layer error:

```text
Generic HdfsObjectStore error – RPC listener disconnected
```

The dataset was validated locally before continuing. Because the issue occurred during the Hopsworks storage operation, the validated local dataset was used for the remaining model-development pipeline.

---

## 🖥️ Streamlit Dashboard

The interactive Streamlit dashboard displays:

* Current AQI
* PM2.5
* AQI category
* 72-hour forecast
* Daily forecast summary
* Maximum predicted AQI
* Model performance
* SHAP feature importance
* Individual prediction explanation
* Model information
* Data information
* AQI health alerts
* About section

---

## 🌐 Live OpenWeather Integration

The final dashboard connects directly to the live OpenWeather Air Pollution API.

The application:

1. Requests the latest air-quality data
2. Processes the live observation
3. Calculates the displayed AQI from PM2.5
4. Generates the 72-hour forecast
5. Displays the results through Streamlit

If the live API request fails, the application can fall back to the latest stored observation.

The dashboard is configured to refresh periodically so that updated air-quality information can be displayed.

---

## 🧮 AQI Calculation

OpenWeather provides an AQI value on a scale from **1 to 5**.

This value is not the same as the commonly used EPA AQI scale of 0–500.

Therefore, the project does not directly use the OpenWeather 1–5 value as the displayed AQI.

Instead, the dashboard calculates an **EPA-style AQI from PM2.5 concentration** and displays the corresponding AQI category.

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
* OpenWeather API
* Git
* GitHub

### Note

TensorFlow was explored during development but was **not used as the final production model**.

The final production model is the **Scikit-learn Gradient Boosting Regressor**.

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

### 1. Clone the Repository

```bash
git clone https://github.com/anusha-awan/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file containing the required API credentials:

```text
OPENWEATHER_API_KEY=your_api_key
HOPSWORKS_API_KEY=your_api_key
```

Do not commit `.env` or API keys to GitHub.

### 6. Run the Dashboard

```bash
streamlit run app.py
```

---

## 🔐 Security

API credentials are stored separately from the source code.

During local development, the OpenWeather API key is stored in a `.env` file.

For cloud deployment, the API key is stored using Streamlit Secrets.

The `.env` file is excluded from version control through `.gitignore`.

This prevents sensitive API credentials from being exposed in the public GitHub repository.

---

## ☁️ Hopsworks Integration

Hopsworks was explored for cloud-based machine-learning infrastructure.

The project successfully:

* Connected to Hopsworks
* Created the `aqi_features_v2` Feature Store
* Explored Feature Store functionality
* Explored Model Registry functionality
* Registered model entries

However, the final Feature Store data upload failed with:

```text
Generic HdfsObjectStore error – RPC listener disconnected
```

Because the local dataset had already been validated successfully, development continued using the validated local dataset.

The final production model was therefore trained using the local `features.csv` dataset rather than directly retrieving training data from Hopsworks.

---

## 🧪 Model Registry

Hopsworks Model Registry functionality was also explored.

Model entries included:

* `aqi_random_forest`
* `aqi_gradient_boosting`
* `pearls_aqi_predictor`

The Model Registry was used to explore model management and versioning.

Some model versions differed during development, so registry/version information should be interpreted as part of the project's development history.

---

## 📌 Project Deliverables

The completed project provides:

* End-to-end AQI prediction application
* Real-time air-quality API integration
* Historical data collection
* Historical feature processing
* Machine-learning model comparison
* Gradient Boosting production model
* 72-hour AQI forecasting
* SHAP model explainability
* AQI health alerts
* Streamlit dashboard
* GitHub Actions automation
* Hopsworks Feature Store exploration
* Hopsworks Model Registry exploration
* Secure API-key handling
* Online deployment
* Project documentation

---

## ⚠️ Current Limitations

The current project has several documented limitations:

### Hopsworks Data Upload

The Hopsworks Feature Store was created successfully, but the final data upload failed because of a storage-layer RPC error.

### Local Training Dataset

The final production model was trained using the validated local dataset rather than directly retrieving training data from Hopsworks.

### Flask/FastAPI Backend

A separate Flask/FastAPI REST backend was not implemented in the final version.

Streamlit currently acts as the application and presentation layer.

### TensorFlow

TensorFlow was explored but was not selected as the final production framework.

### Location

The current model is designed for Lahore, Pakistan.

### API Dependency

The live dashboard depends on the availability of the OpenWeather API.

### Forecast Uncertainty

Predicted AQI values are estimates and actual future air quality can differ because environmental conditions can change unexpectedly.

---

## 🚀 Future Improvements

Possible future improvements include:

* Add temperature as a model feature
* Add humidity
* Add wind speed and wind direction
* Add atmospheric pressure
* Add more historical air-quality data
* Support multiple cities
* Add more time-series models
* Add advanced deep-learning forecasting
* Add a Flask/FastAPI backend
* Connect model training directly with the Feature Store
* Resolve the Hopsworks storage issue
* Improve model version management
* Add automatic model retraining
* Add prediction monitoring
* Add email or notification alerts
* Add AQI health recommendations
* Add uncertainty ranges
* Improve mobile responsiveness
* Add detailed historical AQI visualizations

---

## 👩‍💻 Project

**Pearls AQI Predictor**

Built as part of the **10Pearls SHINE – Data Sciences Internship** project.

**Developer:** Anusha Awan
**Program:** BS Software Engineering
**Institution:** Government College University, Lahore

---

## 🔗 Project Links

### GitHub Repository

https://github.com/anusha-awan/pearls-aqi-predictor

### Live Streamlit Dashboard

https://pearlsaqi2026.streamlit.app/
