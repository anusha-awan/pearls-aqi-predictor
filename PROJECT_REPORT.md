# PEARLS AQI PREDICTOR

## DATA SCIENCE INTERNSHIP PROJECT REPORT

**Submitted by:** Anusha Awan  
**Program:** BS Software Engineering  
**Institution:** Government College University, Lahore  
**Internship:** 10Pearls SHINE – Data Sciences  
**Project Submission:** September 2026

---

## 1. Project Overview

**Project Name:** Pearls AQI Predictor  
**Location:** Lahore, Pakistan  
**Project Type:** Machine Learning / Data Science  

The goal of this project is to predict the Air Quality Index (AQI) of Lahore for the next three days using air-quality data and machine learning.

The project collects air-quality data, processes it, creates useful features, trains different machine learning models, selects the best model, and uses it to predict future AQI values.

A Streamlit dashboard was also created so that the results can be viewed in a simple and user-friendly way.

---

## 2. Problem Statement

Air pollution is an important problem in Lahore. Air quality can change because of different pollutants and environmental conditions.

The main problem addressed in this project is:

**Can machine learning be used to predict the AQI of Lahore for the next three days using historical and current air-quality data?**

The purpose is not only to show the current AQI but also to estimate future AQI values so that users can get an idea of upcoming air quality.

---

## 3. Project Objectives

The main objectives of the project were:

- Collect air-quality data for Lahore.
- Use an external air-quality API.
- Store and prepare historical data.
- Clean the collected data.
- Create useful features for machine learning.
- Calculate AQI using PM2.5 values.
- Train different machine learning models.
- Compare model performance.
- Select the best-performing model.
- Predict AQI for the next 72 hours.
- Add explainability using SHAP.
- Try to use Hopsworks Feature Store and Model Registry.
- Automate data collection and project tasks using GitHub Actions.
- Build an interactive Streamlit dashboard.
- Connect the dashboard with live air-quality data.
- Deploy the dashboard online.

---

## 4. Initial Project Requirements

At the beginning, the project required the use of several technologies and tools.

The main technologies considered were:

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- OpenWeather API / AQICN
- Hopsworks or Vertex AI
- Apache Airflow or GitHub Actions
- Streamlit
- Flask or FastAPI
- SHAP
- Git and GitHub

The project was planned as a serverless-style data science application where external APIs, cloud services, automation, machine learning and a web dashboard would work together.

---

## 5. Planned Architecture

The original idea was to create a complete pipeline:

**Air Quality API → Data Collection → Data Cleaning → Feature Engineering → Feature Store → Model Training → Model Registry → Prediction → Dashboard**

The system was divided into different parts.

### Data Source

OpenWeather API was used to collect air-quality information for Lahore.

### Data Processing

Python was used for:

- Collecting data
- Cleaning data
- Creating features
- Calculating AQI
- Preparing datasets

### Machine Learning

Different machine learning models were trained and compared.

### Model Management

Hopsworks was explored for:

- Feature Store
- Model Registry

### Automation

GitHub Actions was used for automation.

### Application

Streamlit was used to create the final dashboard.

---

## 6. Development Journey

The project was not completed in one step. It was developed gradually.

Different problems appeared during development, so the project was tested and improved step by step.

The major stages were:

1. Setting up the Python environment
2. Connecting the API
3. Collecting data
4. Creating historical data
5. Cleaning and preparing the dataset
6. Feature engineering
7. AQI calculation
8. Training machine learning models
9. Comparing models
10. Creating predictions
11. Trying Hopsworks
12. Working with the Model Registry
13. Adding SHAP
14. Creating EDA
15. Creating the Streamlit dashboard
16. Connecting live API data
17. Adding dashboard customization
18. Adding automation
19. Deploying the dashboard
20. Testing the final system

---

## 7. Python Environment Setup

Python was used as the main programming language.

A virtual environment was created so that project dependencies could be installed separately from the main Python installation.

The project used **Python 3.14.3**.

Important libraries used in the project included:

- pandas
- numpy
- scikit-learn
- requests
- python-dotenv
- joblib
- Streamlit
- SHAP
- matplotlib

TensorFlow was also explored as part of the project requirements and experiments.

---

## 8. OpenWeather API Integration

One of the first important steps was connecting the project with a real air-quality API.

OpenWeather Air Pollution API was selected.

The API was used to collect air-quality information for Lahore using its latitude and longitude.

The project used:

- **Latitude:** 31.5204
- **Longitude:** 74.3587

The API provides information about pollutants such as:

- CO
- NO
- NO₂
- O₃
- SO₂
- PM2.5
- PM10
- NH₃

The API also provides an OpenWeather AQI value.

### OpenWeather AQI vs EPA AQI

However, an important point was discovered during development.

The AQI value provided by OpenWeather uses a scale from **1 to 5**. This is not the same as the commonly used EPA AQI scale of **0 to 500**.

Therefore, the project does not directly display the OpenWeather AQI value as the final AQI.

Instead, the project calculates the displayed EPA-style AQI from PM2.5 values.

This was an important design decision because using the OpenWeather 1–5 value directly would have produced a misleading AQI display.

---

## 9. API Key Security

The API key was not written directly inside the source code.

A `.env` file was used during local development.

The application loads the API key using environment variables.

For the deployed Streamlit application, Streamlit Secrets were used.

This keeps the API key separate from the public GitHub repository.

This was especially important because the GitHub repository is public.

---

## 10. Data Collection

A Python script named `save_data.py` was created for collecting air-quality data.

The script:

1. Loads the API key.
2. Sends a request to OpenWeather.
3. Gets the latest air-quality information.
4. Extracts the required fields.
5. Saves the current observation.
6. Adds the observation to the historical dataset.

The collected information includes:

- timestamp
- AQI-related information
- PM2.5
- PM10
- CO
- NO
- NO₂
- O₃
- SO₂
- NH₃

The data was stored locally in CSV files.

---

## 11. Historical Data

Historical data was required for machine learning.

The project created a historical dataset by collecting and storing observations over time.

The historical dataset was then used for:

- Analysis
- Feature engineering
- Model training
- Testing

The final validated local dataset contained:

- **8,401 valid rows**
- **28 features**

The dataset was checked for common data problems.

The validation showed:

- No missing feature values
- No duplicate datetime values

This validation was important before moving towards model training and cloud storage.

---

## 12. Data Cleaning

Before training the models, the data was checked and prepared.

The cleaning process included:

- Checking missing values
- Checking duplicate timestamps
- Converting timestamps into the correct format
- Arranging data in chronological order
- Selecting useful columns
- Preparing numerical features

The goal was to make sure that the machine learning models received clean and consistent data.

---

## 13. Feature Engineering

Feature engineering was one of the important parts of the project.

Raw air-quality values alone were not enough. Additional features were created from the available data.

These features helped the machine learning models understand patterns in air quality over time.

Examples of useful information included:

- Pollutant values
- Previous AQI values
- Historical/lagged values
- Time-related information
- Rolling information
- Other derived numerical features

The final production model used **26 features**.

Feature engineering was also important for the 72-hour recursive forecasting process.

---

## 14. AQI Calculation

The dashboard calculates the displayed AQI from PM2.5.

PM2.5 is one of the most important pollutants for AQI calculation.

The project uses PM2.5 concentration and maps it to the corresponding AQI range.

This allows the dashboard to display:

- AQI value
- AQI category
- PM2.5 concentration

instead of simply showing the 1–5 OpenWeather AQI scale.

---

## 15. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset before relying on the machine learning results.

The analysis focused on:

- Pollutant values
- AQI patterns
- Relationships between features
- Data distribution
- Historical trends

EDA helped in understanding the data and selecting useful features for the model.

It also provided a better understanding of the air-quality dataset before model training.

---

## 16. Machine Learning Models

Multiple machine learning models were trained instead of using only one model.

The main models compared were:

- Random Forest
- Gradient Boosting
- Ridge Regression

The purpose of model comparison was to determine which model produced the best predictions.

Three main evaluation metrics were used.

### MAE

Mean Absolute Error shows the average absolute difference between the actual and predicted values.

### RMSE

Root Mean Squared Error gives more weight to larger prediction errors.

### R²

R² shows how well the model explains the variation in the target values.

---

## 17. Model Comparison

The following results were obtained during model evaluation.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 3.6747 | 6.8945 | 0.9433 |
| Gradient Boosting | 3.5714 | 6.0362 | 0.9565 |
| Ridge Regression | 4.9105 | 7.3455 | 0.9356 |

Gradient Boosting produced the best overall results.

It had:

- Lowest MAE
- Lowest RMSE
- Highest R²

Therefore, Gradient Boosting was selected as the main production model.

---

## 18. Final Model

The final selected model was:

**Gradient Boosting Regressor**

The model uses **26 features**.

It is used to generate predictions for the next **72 hours**.

The prediction process is recursive.

This means that predicted values are used as part of the input when generating later future predictions.

The model was selected based on its evaluation results rather than simply choosing a model without comparison.

---

## 19. Prediction Testing

The prediction script was tested separately.

The system successfully generated a 72-hour forecast.

In an earlier prediction test, the predicted AQI values were approximately:

- **Minimum:** 90.7
- **Maximum:** 109.7
- **Average:** 102.5

The exact forecast changes when fresh data is used.

This is expected because the model is designed to work with changing air-quality data.

---

## 20. Hopsworks Feature Store

Hopsworks was explored as part of the project requirements.

The Feature Store was successfully connected and the feature store named:

`aqi_features_v2`

was created.

The local dataset was also validated before attempting the final upload.

The validation showed that the dataset was not failing because of missing values or duplicate timestamps.

However, a problem occurred during the final storage/upload operation.

The upload failed with an error related to the Hopsworks storage layer:

`Generic HdfsObjectStore error – RPC listener disconnected`

This happened during the storage operation rather than during dataset preparation.

Because the data itself was already validated locally, the issue was treated as a cloud/storage-side problem rather than a data-quality problem.

---

## 21. Alternative Path After Hopsworks Issue

Because the Hopsworks upload could not be completed successfully, the validated local dataset was kept as the working dataset for the remaining development.

The project was not stopped because of the cloud storage problem.

The remaining components were completed using the working local pipeline.

This allowed the following parts to continue:

- Model training
- Model comparison
- Prediction
- SHAP
- Dashboard development
- Live API integration
- Deployment

The upload script was also adjusted so that this known storage problem would not unnecessarily stop the rest of the pipeline.

### Important Limitation

The final project should not claim that the complete dataset was successfully uploaded and committed to Hopsworks.

The correct statement is that Hopsworks was successfully connected and the Feature Store was created, but the final storage/upload operation failed.

---

## 22. Hopsworks Model Registry

Model Registry functionality was also explored.

Models were registered during the project.

The registry contained model entries including:

- `aqi_random_forest`
- `aqi_gradient_boosting`
- `pearls_aqi_predictor`

The Model Registry was used to demonstrate model management and versioning.

There is a version difference between some registry entries and the dashboard metadata, so this should be treated as a documentation/versioning limitation rather than claiming that every displayed version refers to the same registry entry.

---

## 23. TensorFlow

TensorFlow was included in the project requirements and was explored through the project experiments.

However, the final production model is not a TensorFlow model.

The final production model is the Scikit-learn Gradient Boosting model because it produced the best evaluation results among the tested production approaches.

Therefore, TensorFlow should be described as an explored/experimental technology rather than the final production model.

---

## 24. SHAP Explainability

SHAP was added to make the machine learning results easier to understand.

Machine learning models can produce predictions without clearly explaining why a prediction was made.

SHAP helps show the importance and contribution of features.

The dashboard includes SHAP-based model explainability information.

This gives users a better idea of which features have more influence on the model predictions.

---

## 25. Automation with GitHub Actions

GitHub Actions was used for automation.

The purpose was to reduce the need to manually run project tasks every time.

The workflow can be used for automated project operations such as data collection and pipeline execution.

This was useful because an AQI prediction system should be able to work with updated data instead of depending completely on manual execution.

---

## 26. Streamlit Dashboard

A Streamlit dashboard was created as the main user interface.

The dashboard provides a simple way to view the project results.

The dashboard includes:

- Current AQI
- AQI category
- PM2.5
- 3-day prediction
- Maximum predicted AQI
- Model performance
- SHAP information
- Data/model information
- About section

The dashboard was designed to keep the results understandable for a normal user instead of showing only technical model output.

---

## 27. Live OpenWeather Integration

One of the important improvements made near the end of the project was connecting the dashboard directly to the live OpenWeather API.

Previously, the dashboard could use stored data.

The final version was improved so that it first tries to get fresh data from OpenWeather.

The application uses:

**Streamlit Secrets → OpenWeather API → Latest Air Quality → AQI Calculation → Prediction**

If the API request fails, the application can fall back to the latest stored observation.

This makes the dashboard more reliable while still allowing it to display stored data if a temporary API problem occurs.

---

## 28. Live Data Refresh

The dashboard displays a message showing that live air-quality data is being used.

The dashboard was configured to refresh the data periodically.

The current implementation displays a refresh interval of approximately five minutes.

This means the dashboard can use updated API information instead of depending only on an old CSV observation.

---

## 29. Final Dashboard Example

During final local testing, the dashboard successfully showed:

**Live air-quality data from OpenWeather API**

**Latest observation:** 2026-09-03 14:31

**Current EPA AQI:** 113

**AQI Category:** Unhealthy for Sensitive Groups

**PM2.5:** 40.24

### Three-Day Forecast

- Day 1: 114.7
- Day 2: 114.7
- Day 3: 114.7

**Maximum predicted AQI:** 121.8

These values are examples from one live run.

Because the project uses live API data, the values can change with new observations.

---

## 30. Dashboard Design

The dashboard was customized instead of using the default Streamlit appearance.

A light cream and brown theme was selected.

The main design choices included:

- Cream background
- Brown primary color
- Dark brown text
- Serif font
- Soft neutral colors

The goal was to make the dashboard look clean and different from a basic default Streamlit application.

The theme was also configured through Streamlit's theme settings instead of adding unnecessary custom CSS.

---

## 31. User Interface Features

The dashboard was kept simple so that users can quickly understand the results.

The main information is presented through sections such as:

### Current Air Quality

Shows the latest available AQI and pollutant information.

### AQI Forecast

Shows the predicted AQI for the next three days.

### Model Performance

Shows the performance of the trained machine learning models.

### Explainability

Shows SHAP information to help understand the model.

### Data and Model Information

Provides information about the dataset and model used.

### About

Explains the purpose of the Pearls AQI Predictor.

---

## 32. Application Serving Layer

Streamlit was used as the main application and presentation layer.

The dashboard directly loads the trained model, processes the latest air-quality data, generates the forecast, and displays the results.

A separate Flask or FastAPI REST backend was not implemented in the final version.

This is an important limitation because Flask/FastAPI was part of the original project requirements.

The decision was made to keep the working Streamlit application stable instead of introducing another backend layer without confirmation that it was mandatory.

---

## 33. Git and GitHub

Git was used for version control.

The project was maintained in a GitHub repository.

**Repository:**  
https://github.com/anusha-awan/pearls-aqi-predictor

Git was used to:

- Track changes
- Commit updates
- Push project files
- Manage the project repository
- Support deployment

Sensitive API keys were kept outside the public source code.

---

## 34. Deployment

The Streamlit application was deployed online.

**Live Streamlit Dashboard:**  
https://pearlsaqi2026.streamlit.app/

The deployed version uses Streamlit Secrets for the OpenWeather API key.

This allows the public dashboard to request live data without exposing the API key in the GitHub repository.

---

## 35. Testing

Testing was performed at different stages of the project.

### API Testing

The OpenWeather API was tested separately.

Initially, an API authentication problem occurred.

The API returned:

`401 – Invalid API Key`

After checking the API configuration and key, the API started returning:

`200 – Successful response`

The returned JSON was then checked to make sure the required pollutant values were available.

---

## 36. Local Application Testing

The Streamlit dashboard was tested locally before deployment.

During testing, the dashboard successfully:

- Loaded the application
- Accessed the API key
- Requested air-quality data
- Calculated AQI
- Generated predictions
- Displayed model information
- Displayed SHAP information

A missing `.env` loading step was identified during testing.

The problem was fixed by loading environment variables before reading the API key.

---

## 37. Cloud Deployment Testing

The deployed application was also tested after adding the OpenWeather API key to Streamlit Secrets.

The API key was added using the correct TOML format.

This allowed the cloud application to access the API without placing the secret directly inside the source code.

The deployed dashboard was then checked for live data and updated timestamps.

---

## 38. Problems Faced During Development

Several problems were faced during the project.

| Problem | Solution |
|---|---|
| OpenWeather returned 401 invalid API key | Checked API key configuration and corrected the setup |
| Dashboard could not read the local API key | Added `.env` loading using `load_dotenv()` |
| OpenWeather AQI was only 1–5 | Used PM2.5 to calculate the displayed EPA-style AQI |
| Hopsworks upload failed | Validated data locally and continued with the working local pipeline |
| Hopsworks storage returned RPC listener error | Treated it as a storage-side problem instead of a dataset problem |
| Cloud dashboard needed API key | Added the key through Streamlit Secrets |
| Public repository could expose secrets | Kept the API key in `.env` / Streamlit Secrets |
| Different model versions existed during development | Kept model registry information documented and noted the version limitation |
| Dashboard initially depended on stored data | Added live OpenWeather API integration |
| Too many unnecessary UI changes could create problems | Used Streamlit theme configuration instead of adding complex custom CSS |

---

## 39. Important Design Decisions

Several decisions were made during development.

### Decision 1: Use OpenWeather

OpenWeather was selected because it provided the required air-quality pollutant information through an accessible API.

### Decision 2: Calculate AQI from PM2.5

The OpenWeather 1–5 AQI scale was not treated as the final EPA AQI.

PM2.5 was used for the dashboard's AQI calculation.

### Decision 3: Use Gradient Boosting

Multiple models were compared.

Gradient Boosting produced the best results, so it was selected as the production model.

### Decision 4: Keep a Local Fallback

The dashboard was designed to use stored data if the live API request fails.

This makes the application more robust.

### Decision 5: Use Streamlit

Streamlit allowed the complete prediction system to be shown in an interactive dashboard without requiring a separate frontend application.

### Decision 6: Keep the Hopsworks Limitation Honest

The Hopsworks connection and Feature Store creation worked, but the final storage upload did not.

Instead of claiming success, the project continued using the validated local pipeline.

---

## 40. Current System Flow

The final working flow is:

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
