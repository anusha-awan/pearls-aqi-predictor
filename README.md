# 🌍 Pearls AQI Predictor

An end-to-end machine learning application that predicts **Air Quality Index (AQI) for the next 3 days** using historical air-quality data, automated feature engineering, and a Random Forest regression model.

The project includes model training, chronological evaluation, automated feature processing, GitHub Actions integration, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Air quality can change significantly over time due to changes in particulate matter and atmospheric pollutants.

**Pearls AQI Predictor** uses historical pollution data to learn AQI patterns and generate hourly AQI predictions for the next **72 hours**.

### Main objectives

* Collect and process historical air-quality data
* Engineer time-series and lag-based features
* Calculate an EPA-style AQI target from PM2.5
* Compare multiple machine-learning models
* Select the best-performing model
* Evaluate the model using a chronological train-test split
* Generate a 72-hour AQI forecast
* Present predictions through an interactive Streamlit dashboard
* Automate feature processing using GitHub Actions

---

## 🏗️ System Architecture

```text
Air Quality Data
       │
       ▼
Data Collection
       │
       ▼
Feature Engineering
       │
       ├── Pollutant Features
       ├── Time Features
       ├── Lag Features
       └── Rolling Features
       │
       ▼
EPA-style PM2.5 AQI Calculation
       │
       ▼
Training Dataset
       │
       ▼
Chronological 80/20 Split
       │
       ▼
Model Training
       │
       ├── Ridge Regression
       ├── Random Forest
       └── Gradient Boosting
       │
       ▼
Model Comparison
       │
       ▼
Random Forest Selected
       │
       ▼
Saved Model (aqi_model.pkl)
       │
       ▼
Streamlit Dashboard
       │
       ├── Current AQI
       ├── 72-Hour Forecast
       ├── Daily Forecast
       ├── Model Performance
       ├── Feature Importance
       └── AQI Health Alerts
```

---

## 🧪 Data and Features

The project uses air-quality observations containing pollutant measurements such as:

* CO
* NO
* NO₂
* O₃
* SO₂
* PM2.5
* PM10
* NH₃

Additional engineered features include:

### Time features

* Hour
* Day
* Month
* Day of week

### AQI lag features

* AQI lag 1
* AQI lag 3
* AQI lag 6
* AQI lag 12
* AQI lag 24
* AQI lag 48
* AQI lag 72

### Pollution lag features

* PM2.5 lag 1
* PM10 lag 1

### Rolling AQI features

* 6-hour rolling AQI
* 24-hour rolling AQI
* 72-hour rolling AQI

### Additional feature

* AQI change

The final model uses **26 input features**.

---

## 📐 EPA-style AQI Target

The project calculates an **EPA-style AQI from PM2.5 concentration** rather than using the original 1–5 air-quality index supplied by the source API.

The resulting AQI follows the standard **0–500 AQI scale**.

The calculated AQI is shifted by one hour to create the prediction target:

```text
Current observations → Next-hour AQI target
```

This prevents the model from simply predicting the AQI value at the same timestamp.

---

## 🤖 Machine Learning Models

Three regression models were evaluated:

1. Ridge Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

A chronological **80/20 train-test split** was used instead of randomly shuffling the time-series data.

### Model comparison

| Model             |      MAE |     RMSE |         R² |
| ----------------- | -------: | -------: | ---------: |
| Ridge Regression  |    11.78 |    14.40 |     0.7519 |
| Random Forest     | **3.72** | **6.58** | **0.9483** |
| Gradient Boosting |     4.64 |     7.02 |     0.9411 |

### Selected model

**Random Forest Regressor**

Selection criterion:

**Lowest RMSE on the chronological test set.**

---

## 📊 Model Evaluation

Final test-set performance:

* **MAE:** 3.72 AQI points
* **RMSE:** 6.58 AQI points
* **R²:** 0.9483

The evaluation was performed on the actual EPA-style AQI scale rather than the original 1–5 source index.

The test-set AQI values ranged from approximately **59 to 210**, while model predictions ranged from approximately **59 to 219**.

---

## 🔮 72-Hour AQI Forecast

The Streamlit application generates hourly predictions for the next **72 hours**.

The dashboard provides:

* 72-hour AQI prediction chart
* Daily minimum AQI
* Daily average AQI
* Daily maximum AQI
* AQI health alerts
* Current AQI
* Current PM2.5 concentration

The forecast is divided into three daily periods for easier interpretation.

---

## 🖥️ Streamlit Dashboard

The dashboard provides an interactive interface containing:

### Current Air Quality

Displays:

* Current EPA AQI
* AQI category
* PM2.5 concentration
* Latest available timestamp

### Next 3 Days Forecast

Displays:

* Hourly 72-hour forecast
* Forecast graph
* Daily AQI summary

### Model Performance

Displays:

* MAE
* RMSE
* R²

### Model Explainability

Displays Random Forest feature importance through:

* Feature importance table
* Feature importance chart

### AQI Alerts

The dashboard provides warnings when predicted AQI reaches unhealthy levels.

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Matplotlib
* Joblib
* Streamlit
* Hopsworks
* GitHub Actions
* Python dotenv

---

## 📂 Project Structure

```text
pearls-aqi-predictor/
│
├── app.py
├── prepare_training_data.py
├── train_model.py
├── evaluate_model.py
├── predict_3_days.py
│
├── features.csv
├── training_data.csv
├── historical_aqi.csv
│
├── aqi_model.pkl
├── model_comparison.csv
├── model_metadata.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/anusha-awan/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open locally in the browser.

---

## 🔄 Model Training

To prepare the training dataset:

```bash
python prepare_training_data.py
```

To train and compare the models:

```bash
python train_model.py
```

To evaluate the saved model:

```bash
python evaluate_model.py
```

The best model is saved as:

```text
aqi_model.pkl
```

---

## ☁️ Hopsworks Integration

Hopsworks was used as part of the project's feature-store and model-management workflow.

The project was also designed to support loading features and models through Hopsworks.

Due to compute/resource limitations, the final dashboard can run locally using the saved model while the automated pipeline remains integrated with the project workflow.

The Hopsworks compute limitation is documented as a project limitation rather than treated as a model failure.

---

## 🔁 GitHub Actions

GitHub Actions is used to automate the AQI feature pipeline.

The feature pipeline successfully completed execution on GitHub Actions.

The workflow processes the required feature-engineering pipeline automatically and supports reproducibility of the project workflow.

---

## ⚠️ Limitations

The current implementation has several limitations:

1. The forecasting model predicts AQI based on currently available pollutant and engineered features.
2. Future pollutant concentrations are not independently forecasted.
3. The 72-hour recursive forecast therefore relies on the latest available feature state together with future time features.
4. The current system focuses on PM2.5-based EPA-style AQI.
5. Hopsworks compute availability may limit cloud-based execution.
6. The current dataset represents a limited historical period and additional data could improve generalization.

These limitations should be considered when interpreting long-horizon forecasts.

---

## 🔮 Future Improvements

Future versions could improve the system by:

* Forecasting future pollutant concentrations separately
* Using weather variables such as temperature, humidity, wind speed, and pressure
* Adding more historical data
* Testing XGBoost or other advanced time-series models
* Adding prediction intervals
* Implementing automated model retraining
* Deploying the complete pipeline to a cloud environment
* Adding location-based AQI forecasting
* Supporting multiple pollutant-specific AQI calculations

---

## 📌 Key Results

The final system achieved:

```text
Model: Random Forest Regressor

MAE  : 3.72 AQI points
RMSE : 6.58 AQI points
R²   : 0.9483

Forecast Horizon: 72 hours
Input Features: 26
AQI Scale: 0–500
```

---

## 👩‍💻 Project

**Pearls AQI Predictor**

Developed as an end-to-end machine-learning project integrating data processing, feature engineering, model training, evaluation, automation, and interactive visualization.
