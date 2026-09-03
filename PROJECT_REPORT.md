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

The project collects air-quality data, processes and cleans the data, creates useful features, trains multiple machine learning models, evaluates their performance, selects the best-performing model, and uses it to generate future AQI predictions.

A Streamlit dashboard was also developed to present the current air quality, predicted AQI values, model information, and explainability results in a simple and user-friendly interface.

---

## 2. Problem Statement

Air pollution is an important environmental problem in Lahore. Air quality can change because of different pollutants and environmental conditions.

The main problem addressed by this project is:

**Can machine learning be used to predict the AQI of Lahore for the next three days using historical and current air-quality data?**

The purpose of the project is not only to display the current AQI but also to estimate future AQI values so that users can get an indication of upcoming air-quality conditions.

---

## 3. Project Objectives

The main objectives of the project were:

* Collect air-quality data for Lahore.
* Use an external air-quality API.
* Store and prepare historical data.
* Clean the collected data.
* Create useful features for machine learning.
* Calculate AQI using PM2.5 values.
* Train different machine learning models.
* Compare model performance.
* Select the best-performing model.
* Predict AQI for the next 72 hours.
* Add model explainability using SHAP.
* Explore Hopsworks Feature Store and Model Registry.
* Automate data collection and project tasks using GitHub Actions.
* Build an interactive Streamlit dashboard.
* Connect the dashboard with live air-quality data.
* Deploy the dashboard online.

---

## 4. Initial Project Requirements

At the beginning of the project, several technologies and tools were considered as part of the requirements.

The main technologies considered were:

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* OpenWeather API / AQICN
* Hopsworks or Vertex AI
* Apache Airflow or GitHub Actions
* Streamlit
* Flask or FastAPI
* SHAP
* Git and GitHub

The project was initially planned as a serverless-style data science application in which external APIs, cloud services, automation, machine learning, and a web dashboard would work together.

---

## 5. Planned Architecture

The original plan was to create a complete pipeline:

**Air Quality API → Data Collection → Data Cleaning → Feature Engineering → Feature Store → Model Training → Model Registry → Prediction → Dashboard**

The system was divided into several components.

### Data Source

OpenWeather API was used to collect air-quality information for Lahore.

### Data Processing

Python was used for:

* Collecting data
* Cleaning data
* Creating features
* Calculating AQI
* Preparing datasets

### Machine Learning

Multiple machine learning models were trained and compared to determine the best-performing approach.

### Model Management

Hopsworks was explored for:

* Feature Store
* Model Registry

### Automation

GitHub Actions was used to automate selected project tasks.

### Application

Streamlit was used to create the final interactive dashboard.

---

## 6. Development Journey

The project was developed gradually rather than being completed in a single step.

Different technical problems appeared during development, so each major component was tested and improved before moving to the next stage.

The major development stages were:

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
11. Exploring Hopsworks
12. Working with the Model Registry
13. Adding SHAP
14. Performing EDA
15. Creating the Streamlit dashboard
16. Connecting the live API
17. Customizing the dashboard
18. Adding automation
19. Deploying the dashboard
20. Testing the final system

---

## 7. Python Environment Setup

Python was used as the main programming language throughout the project.

A virtual environment was created so that project dependencies could be installed separately from the main Python installation.

The project used **Python 3.14.3**.

Important libraries used in the project included:

* pandas
* numpy
* scikit-learn
* requests
* python-dotenv
* joblib
* Streamlit
* SHAP
* matplotlib

TensorFlow was also explored as part of the project requirements and experiments.

---

## 8. OpenWeather API Integration

One of the first major steps was connecting the project to a real air-quality API.

The OpenWeather Air Pollution API was selected as the main data source.

The API was used to collect air-quality information for Lahore using its latitude and longitude.

The project used:

* **Latitude:** 31.5204
* **Longitude:** 74.3587

The API provides information about pollutants such as:

* CO
* NO
* NO₂
* O₃
* SO₂
* PM2.5
* PM10
* NH₃

The API also provides an OpenWeather AQI value.

### OpenWeather AQI vs EPA AQI

An important difference was identified during development.

The AQI value provided by OpenWeather uses a scale from **1 to 5**. This is different from the commonly used EPA AQI scale of **0 to 500**.

Therefore, the project does not directly display the OpenWeather 1–5 value as the final AQI.

Instead, the project calculates the displayed EPA-style AQI from PM2.5 concentration.

This was an important design decision because directly presenting the OpenWeather 1–5 value as an EPA-style AQI could result in a misleading display.

---

## 9. API Key Security

The API key was not hard-coded into the source code.

During local development, a `.env` file was used to store the API key.

The application loads the key through environment variables.

For the deployed Streamlit application, Streamlit Secrets was used to securely provide the API key.

This keeps the API key separate from the public GitHub repository and reduces the risk of exposing sensitive credentials.

---

## 10. Data Collection

A Python script named `save_data.py` was created to collect air-quality data.

The script:

1. Loads the API key.
2. Sends a request to OpenWeather.
3. Retrieves the latest air-quality information.
4. Extracts the required fields.
5. Saves the current observation.
6. Adds the observation to the historical dataset.

The collected information includes:

* timestamp
* AQI-related information
* PM2.5
* PM10
* CO
* NO
* NO₂
* O₃
* SO₂
* NH₃

The data was stored locally in CSV files.

---

## 11. Historical Data

Historical data was required for machine learning and analysis.

The project created a historical dataset by collecting and storing observations over time.

The historical dataset was used for:

* Exploratory analysis
* Feature engineering
* Model training
* Testing

The final validated local dataset contained:

* **8,401 valid rows**
* **28 columns**

The dataset was checked for common data-quality issues.

The validation showed:

* No missing feature values
* No duplicate datetime values

This validation was performed before model training and before attempting cloud-based feature storage.

---

## 12. Data Cleaning

Before training the models, the collected data was checked and prepared.

The cleaning process included:

* Checking for missing values
* Checking for duplicate timestamps
* Converting timestamps into the correct format
* Arranging data in chronological order
* Selecting useful columns
* Preparing numerical features

The objective was to ensure that the machine learning models received clean and consistent data.

---

## 13. Feature Engineering

Feature engineering was an important part of the project.

Raw air-quality measurements alone were not considered sufficient to capture the temporal patterns in the data. Additional features were therefore created from the available observations.

These features helped the machine learning models identify relationships and patterns in air quality over time.

Examples of information used for feature engineering included:

* Pollutant values
* Previous AQI values
* Historical and lagged values
* Time-related information
* Rolling information
* Other derived numerical features

The final production model used **26 features**.

Feature engineering was also important for the 72-hour recursive forecasting process.

---

## 14. AQI Calculation

The dashboard calculates the displayed AQI using PM2.5 concentration.

PM2.5 is one of the important pollutants used in AQI calculations.

The project maps the PM2.5 concentration to the corresponding AQI range so that the dashboard can display:

* AQI value
* AQI category
* PM2.5 concentration

This approach allows the dashboard to provide an EPA-style AQI display instead of directly presenting the OpenWeather 1–5 AQI scale.

---

## 15. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset before relying on the machine learning results.

The analysis focused on:

* Pollutant values
* AQI patterns
* Relationships between features
* Data distribution
* Historical trends

EDA helped in understanding the characteristics of the data and identifying useful patterns and features for model development.

It also provided a better understanding of the air-quality dataset before model training.

---

## 16. Machine Learning Models

Multiple machine learning models were trained rather than relying on a single algorithm.

The main models compared were:

* Random Forest
* Gradient Boosting
* Ridge Regression

The purpose of model comparison was to determine which model provided the most suitable prediction performance.

Three main evaluation metrics were used.

### MAE

Mean Absolute Error (MAE) represents the average absolute difference between actual and predicted values.

### RMSE

Root Mean Squared Error (RMSE) gives greater weight to larger prediction errors.

### R²

R² represents how well the model explains the variation in the target values.

---

## 17. Model Comparison

The following results were obtained during model evaluation.

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Random Forest     | 3.6747 | 6.8945 | 0.9433 |
| Gradient Boosting | 3.5714 | 6.0362 | 0.9565 |
| Ridge Regression  | 4.9105 | 7.3455 | 0.9356 |

Gradient Boosting produced the best overall results.

It achieved:

* The lowest MAE
* The lowest RMSE
* The highest R²

Therefore, Gradient Boosting was selected as the main production model.

---

## 18. Final Model

The final selected model was:

**Gradient Boosting Regressor**

The model uses **26 features**.

It is used to generate predictions for the next **72 hours**.

The prediction process is recursive. This means that predicted values are incorporated into the input used to generate subsequent future predictions.

The model was selected based on its evaluation results rather than being selected without comparison.

---

## 19. Prediction Testing

The prediction script was tested separately from the dashboard.

The system successfully generated a 72-hour forecast.

In an earlier prediction test, the predicted AQI values were approximately:

* **Minimum:** 90.7
* **Maximum:** 109.7
* **Average:** 102.5

These values represent one prediction run and are not fixed project outputs.

The exact forecast changes when fresh air-quality data is used because the model works with changing input conditions.

---

## 20. Hopsworks Feature Store

Hopsworks was explored as part of the project requirements.

The Hopsworks connection was successfully established, and the Feature Store named:

`aqi_features_v2`

was created.

The local dataset was also validated before attempting the final upload.

The validation showed that the dataset was not failing because of missing values or duplicate timestamps.

However, a problem occurred during the final storage/upload operation.

The upload failed with the following Hopsworks storage-layer error:

`Generic HdfsObjectStore error – RPC listener disconnected`

The error occurred during the storage operation rather than during dataset preparation.

Because the dataset had already been validated locally, the issue was treated as a cloud/storage-side problem rather than a data-quality problem.

---

## 21. Alternative Path After Hopsworks Issue

Because the Hopsworks upload could not be completed successfully, the validated local dataset was retained as the working dataset for the remaining development.

The project was not stopped because of the cloud-storage problem.

The remaining components were completed using the working local pipeline, including:

* Model training
* Model comparison
* Prediction
* SHAP analysis
* Dashboard development
* Live API integration
* Deployment

The upload script was also adjusted so that this known storage problem would not unnecessarily prevent the rest of the pipeline from continuing.

### Important Limitation

The final project does not claim that the complete dataset was successfully uploaded to Hopsworks.

The accurate description is that Hopsworks was successfully connected and the Feature Store was created, but the final storage/upload operation failed because of the reported storage-layer error.

---

## 22. Hopsworks Model Registry

Hopsworks Model Registry functionality was also explored.

Models were registered during the project.

The registry contained model entries including:

* `aqi_random_forest`
* `aqi_gradient_boosting`
* `pearls_aqi_predictor`

The Model Registry was used to demonstrate model management and versioning.

There is a version difference between some registry entries and the dashboard metadata. This is documented as a version-management limitation rather than claiming that every displayed version refers to the same registry entry.

---

## 23. TensorFlow

TensorFlow was included in the original project requirements and was explored during the project experiments.

However, the final production model is not a TensorFlow model.

The final production model is the Scikit-learn Gradient Boosting model because it achieved the best evaluation results among the tested production approaches.

Therefore, TensorFlow is considered an explored or experimental technology rather than the final production technology.

---

## 24. SHAP Explainability

SHAP was added to make the machine learning results easier to interpret.

Machine learning models can generate predictions without directly showing users why a particular prediction was produced.

SHAP helps explain feature importance and the contribution of features to model predictions.

The dashboard includes SHAP-based model explainability information.

This provides users with a better understanding of which features have greater influence on the model predictions.

---

## 25. Automation with GitHub Actions

GitHub Actions was used to automate selected project tasks.

The purpose of automation was to reduce the need to manually execute project operations repeatedly.

The workflows support automated project operations such as data collection and pipeline execution.

This is useful for an AQI prediction system because updated data can be incorporated without relying entirely on manual execution.

---

## 26. Streamlit Dashboard

A Streamlit dashboard was developed as the main user interface of the project.

The dashboard provides a simple way to view the current air-quality information and prediction results.

The dashboard includes:

* Current AQI
* AQI category
* PM2.5
* Three-day prediction
* Maximum predicted AQI
* Model performance
* SHAP information
* Data and model information
* About section

The dashboard was designed to present the results in an understandable way rather than exposing users only to technical model output.

---

## 27. Live OpenWeather Integration

One of the important improvements made near the end of the project was connecting the dashboard directly to the live OpenWeather API.

Previously, the dashboard could use stored data.

The final version was improved so that it first attempts to retrieve fresh data from OpenWeather.

The application follows this process:

**Streamlit Secrets → OpenWeather API → Latest Air Quality → AQI Calculation → Prediction**

If the API request fails, the application can fall back to the latest stored observation.

This makes the dashboard more reliable while still allowing it to display stored information if a temporary API problem occurs.

---

## 28. Live Data Refresh

The dashboard displays a message indicating that live air-quality data is being used.

The dashboard was configured to refresh its data periodically.

The current implementation uses a refresh interval of approximately five minutes.

This allows the dashboard to use updated API information rather than depending only on an older CSV observation.

---

## 29. Final Dashboard Example

During final local testing, the dashboard successfully displayed live air-quality data from the OpenWeather API.

**Latest observation:** 2026-09-03 14:31

**Current EPA AQI:** 113

**AQI Category:** Unhealthy for Sensitive Groups

**PM2.5:** 40.24

### Three-Day Forecast

* Day 1: 114.7
* Day 2: 114.7
* Day 3: 114.7

**Maximum predicted AQI:** 121.8

These values represent one live test run.

Because the application uses live API data, the displayed values can change when new observations become available.

---

## 30. Dashboard Design

The dashboard was customized instead of relying on the default Streamlit appearance.

A light cream and brown theme was selected.

The main design choices included:

* Cream background
* Brown primary color
* Dark brown text
* Serif font
* Soft neutral colors

The objective was to create a clean and distinctive interface rather than using the default Streamlit appearance.

The theme was configured through Streamlit's theme settings instead of adding unnecessary custom CSS.

---

## 31. User Interface Features

The dashboard was designed to remain simple so that users can quickly understand the results.

The main information is presented through the following sections.

### Current Air Quality

Displays the latest available AQI and pollutant information.

### AQI Forecast

Displays the predicted AQI for the next three days.

### Model Performance

Displays the performance of the trained machine learning models.

### Explainability

Displays SHAP-based information to help explain model predictions.

### Data and Model Information

Provides information about the dataset and model used by the application.

### About

Provides a brief explanation of the purpose of the Pearls AQI Predictor project.

---

## 32. Application Serving Layer

Streamlit was used as the main application and presentation layer.

The dashboard directly loads the trained model, processes the latest air-quality data, generates the forecast, and displays the results.

A separate Flask or FastAPI REST backend was not implemented in the final version.

This is an important limitation because Flask/FastAPI was included in the original project requirements.

The decision was made to keep the working Streamlit application stable rather than introduce an additional backend layer without confirmation that it was mandatory.

---

## 33. Git and GitHub

Git was used for version control throughout the project.

The project was maintained in a GitHub repository.

**Repository:**
https://github.com/anusha-awan/pearls-aqi-predictor

Git was used to:

* Track changes
* Commit updates
* Push project files
* Manage the project repository
* Support deployment

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

After checking the API configuration and correcting the key setup, the API started returning:

`200 – Successful response`

The returned JSON was then checked to confirm that the required pollutant values were available.

---

## 36. Local Application Testing

The Streamlit dashboard was tested locally before deployment.

During testing, the dashboard successfully:

* Loaded the application
* Accessed the API key
* Requested air-quality data
* Calculated AQI
* Generated predictions
* Displayed model information
* Displayed SHAP information

A missing `.env` loading step was identified during testing.

The problem was fixed by loading environment variables before reading the API key.

---

## 37. Cloud Deployment Testing

The deployed application was tested after adding the OpenWeather API key to Streamlit Secrets.

The API key was added using the correct TOML format.

This allowed the cloud application to access the API without placing the secret directly inside the source code.

The deployed dashboard was then checked for live data and updated timestamps.

---

## 38. Problems Faced During Development

Several technical problems were encountered during the project.

| Problem                                                          | Solution                                                                    |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------- |
| OpenWeather returned a 401 invalid API key error                 | Checked the API key configuration and corrected the setup                   |
| Dashboard could not read the local API key                       | Added `.env` loading using `load_dotenv()`                                  |
| OpenWeather AQI was provided on a 1–5 scale                      | Used PM2.5 to calculate the displayed EPA-style AQI                         |
| Hopsworks upload failed                                          | Validated the dataset locally and continued with the working local pipeline |
| Hopsworks storage returned an RPC listener error                 | Treated it as a storage-side problem rather than a dataset-quality problem  |
| Cloud dashboard required an API key                              | Added the key through Streamlit Secrets                                     |
| Public repository could expose secrets                           | Kept the API key in `.env` and Streamlit Secrets                            |
| Different model versions existed during development              | Documented the model registry information and version limitation            |
| Dashboard initially depended on stored data                      | Added live OpenWeather API integration                                      |
| Unnecessary UI customization could introduce additional problems | Used Streamlit theme configuration instead of complex custom CSS            |

---

## 39. Important Design Decisions

Several important decisions were made during development.

### Decision 1: Use OpenWeather

OpenWeather was selected because it provided the required air-quality pollutant information through an accessible API.

### Decision 2: Calculate AQI from PM2.5

The OpenWeather 1–5 AQI scale was not treated as the final EPA-style AQI.

PM2.5 was used for the dashboard's AQI calculation.

### Decision 3: Use Gradient Boosting

Multiple models were compared.

Gradient Boosting achieved the best evaluation results and was therefore selected as the production model.

### Decision 4: Keep a Local Fallback

The dashboard was designed to use stored data if the live API request fails.

This improves the robustness of the application.

### Decision 5: Use Streamlit

Streamlit allowed the prediction system to be presented through an interactive dashboard without requiring a separate frontend application.

### Decision 6: Document the Hopsworks Limitation Honestly

The Hopsworks connection and Feature Store creation were successful, but the final storage upload was not.

Instead of claiming complete Hopsworks integration, the project continued using the validated local pipeline and documented the limitation.

---

## 40. Current System Flow

The final working system follows this flow:

**OpenWeather API**
↓
**Live Air-Quality Data**
↓
**Python Data Processing**
↓
**Feature Engineering**
↓
**AQI Calculation**
↓
**Gradient Boosting Model**
↓
**72-Hour Recursive Forecast**
↓
**SHAP Explainability**
↓
**Streamlit Dashboard**
↓
**Online Deployment**

---

## 41. What Is Fully Working

The following components are working in the final project:

* Python environment
* OpenWeather API integration
* Live air-quality data collection
* Historical data handling
* Data cleaning
* Feature engineering
* PM2.5-based AQI calculation
* Machine learning model training
* Model comparison
* Gradient Boosting model
* 72-hour AQI prediction
* SHAP explainability
* Streamlit dashboard
* Live OpenWeather integration
* Streamlit Secrets
* GitHub repository
* GitHub Actions automation
* Online Streamlit deployment

---

## 42. What Has Limitations

Some parts of the original project requirements were not completed exactly as initially planned.

### Hopsworks Data Upload

Hopsworks was successfully connected and the `aqi_features_v2` Feature Store was created. However, the final data upload failed because of a storage-layer error:

`Generic HdfsObjectStore error – RPC listener disconnected`

The dataset was verified locally before continuing. It contained 8,401 valid rows and 28 columns, with no missing feature values and no duplicate datetime values.

Because the problem occurred at the Hopsworks storage layer rather than during data preparation, the validated local dataset was used for the remaining development.

### Flask/FastAPI

A separate Flask/FastAPI backend was not implemented in the final version.

Streamlit currently works as the application and presentation layer. It loads the trained model, processes the latest air-quality data, generates the 72-hour forecast, and displays the results.

### TensorFlow

TensorFlow was explored during the project but was not selected as the final production model.

### Feature Store Training

The final production model was trained using the validated local `features.csv` dataset rather than directly retrieving the training data from Hopsworks.

---

## 43. Model Performance Summary

Three machine learning models were evaluated during the project:

* Random Forest
* Gradient Boosting
* Ridge Regression

The final results were:

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Random Forest     | 3.6747 | 6.8945 | 0.9433 |
| Gradient Boosting | 3.5714 | 6.0362 | 0.9565 |
| Ridge Regression  | 4.9105 | 7.3455 | 0.9356 |

Gradient Boosting achieved the lowest MAE and RMSE and the highest R² score.

Therefore, Gradient Boosting Regressor was selected as the final production model.

---

## 44. Project Strengths

The project has several strengths:

* Uses real air-quality data from an external API.
* Uses live data instead of relying only on static sample data.
* Performs data cleaning and feature engineering.
* Compares multiple machine learning models.
* Selects the final model based on evaluation metrics.
* Provides a 72-hour AQI forecast.
* Includes SHAP-based model explainability.
* Uses GitHub for version control.
* Uses GitHub Actions for automation.
* Provides an interactive Streamlit dashboard.
* Uses secure API-key handling.
* Is deployed online.
* Provides a user-friendly interface.
* Documents technical problems and limitations transparently.

---

## 45. Limitations

The current system has several limitations.

### Limited Location

The current model and dashboard are designed specifically for Lahore, Pakistan.

### API Dependency

The live dashboard depends on the availability and response of the OpenWeather API.

### Forecast Uncertainty

Machine learning predictions are estimates. Actual future AQI values may differ because environmental conditions can change unexpectedly.

### Hopsworks Storage Issue

The Hopsworks Feature Store was created successfully, but the final data upload could not be completed because of the HDFS RPC listener error.

### No Separate REST API

The final version does not include a separate Flask or FastAPI backend.

### Local Training Dataset

The final production model uses the validated local dataset instead of directly retrieving the training data from Hopsworks.

### Limited Environmental Variables

The current model mainly uses air-quality measurements and derived time-based features. Additional weather and environmental variables could improve future predictions.

---

## 46. Future Improvements

The project can be improved in several ways in the future:

* Add temperature as a model feature.
* Add humidity.
* Add wind speed and wind direction.
* Add atmospheric pressure.
* Add more historical air-quality data.
* Support multiple cities.
* Add additional machine learning and time-series models.
* Add advanced deep-learning forecasting models.
* Add a Flask/FastAPI backend.
* Connect model training directly with the Feature Store.
* Resolve the Hopsworks storage issue.
* Improve model version management.
* Add automatic model retraining.
* Add prediction monitoring.
* Add email or notification alerts.
* Add AQI health recommendations.
* Add uncertainty ranges for predictions.
* Improve mobile responsiveness.
* Add more detailed historical AQI visualizations.

---

## 47. Future Flask/FastAPI Architecture

If a separate Flask or FastAPI backend is required, it can be added without rebuilding the complete project.

The future architecture can be:

**OpenWeather API**
↓
**Flask/FastAPI Backend**
↓
**Data Processing**
↓
**Feature Engineering**
↓
**ML Model**
↓
**AQI Prediction**
↓
**Streamlit Dashboard**

In this architecture, the Flask/FastAPI backend would handle prediction-related API requests while Streamlit would primarily focus on the user interface.

The existing model, feature engineering, and prediction logic can be reused.

---

## 48. Security Considerations

Security was considered throughout the development of the project.

The OpenWeather API key was not hard-coded into the public GitHub repository.

During local development, the API key was stored in a `.env` file.

During Streamlit cloud deployment, the API key was stored using Streamlit Secrets.

This approach helps prevent the API key from being exposed in the source code or public repository.

The `.env` file was also kept outside the public project code.

---

## 49. Final Result

The final project successfully provides an end-to-end AQI prediction application for Lahore.

The system can:

* Collect current air-quality data.
* Process the collected data.
* Calculate the current EPA-style AQI from PM2.5.
* Prepare the required model features.
* Load the trained Gradient Boosting model.
* Predict AQI for the next 72 hours.
* Provide model performance information.
* Provide SHAP-based explainability.
* Display current AQI information.
* Display future AQI predictions.
* Present the results through an interactive Streamlit dashboard.
* Run as an online deployed application.

The project also includes GitHub Actions automation and model-management work through Hopsworks.

---

## 50. Lessons Learned

This project helped me understand that developing a data science application involves much more than training a machine learning model.

During the project, I learned about:

* Working with real-world APIs.
* Handling API errors.
* Protecting API keys.
* Collecting historical data.
* Cleaning datasets.
* Feature engineering.
* Calculating AQI from PM2.5.
* Training machine learning models.
* Comparing different models.
* Evaluating model performance.
* Making future predictions.
* Using SHAP for explainability.
* Exploring Feature Stores.
* Registering models.
* Using GitHub Actions.
* Building dashboards with Streamlit.
* Deploying applications online.
* Debugging deployment problems.
* Handling limitations of cloud services.

One of the most important lessons was that real-world projects can encounter technical problems even when the code and data are correct.

The Hopsworks storage issue was an example of this. Instead of stopping the project, I verified the dataset, identified where the problem was occurring, documented the limitation, and continued development using a working alternative.

---

## 51. Conclusion

Pearls AQI Predictor was developed to predict the AQI of Lahore for the next three days using machine learning.

The project started with live air-quality data collection through OpenWeather and gradually developed into an end-to-end data science application involving data processing, feature engineering, AQI calculation, machine learning, model comparison, forecasting, explainability, automation, and deployment.

Three machine learning models were compared, and Gradient Boosting performed the best with an MAE of 3.5714, RMSE of 6.0362, and R² of 0.9565.

The final application uses live OpenWeather data and generates a 72-hour AQI forecast through a recursive prediction approach.

Although some original requirements could not be completed exactly as initially planned, particularly the final Hopsworks data upload and separate Flask/FastAPI backend, these limitations were identified, documented, and handled through alternative approaches.

Overall, this project provided practical experience in developing a real-world machine learning application from data collection through deployment.

---

## 52. Technologies Used

| Technology      | Purpose                                      |
| --------------- | -------------------------------------------- |
| Python          | Main programming language                    |
| Pandas          | Data processing and manipulation             |
| NumPy           | Numerical operations                         |
| Scikit-learn    | Machine learning models                      |
| TensorFlow      | Experimental machine learning exploration    |
| OpenWeather API | Live air-quality data                        |
| Hopsworks       | Feature Store and Model Registry exploration |
| SHAP            | Model explainability                         |
| GitHub Actions  | Automation                                   |
| Streamlit       | Dashboard and application layer              |
| Git             | Version control                              |
| GitHub          | Code repository                              |
| Matplotlib      | Data visualization                           |
| Joblib          | Model saving and loading                     |
| python-dotenv   | Local environment variable management        |

---

## 53. Project Links

### GitHub Repository

https://github.com/anusha-awan/pearls-aqi-predictor

### Live Streamlit Dashboard

https://pearlsaqi2026.streamlit.app/

---

## 54. Final Project Status

**Overall Status:** Completed working prototype with documented limitations

### Working Components

* Python environment
* OpenWeather API
* Live air-quality data collection
* Historical data collection
* Data cleaning
* Feature engineering
* PM2.5-based AQI calculation
* Machine learning model training
* Model comparison
* Gradient Boosting model
* 72-hour recursive forecasting
* SHAP explainability
* Streamlit dashboard
* Live API integration
* Secure API-key handling
* GitHub repository
* GitHub Actions automation
* Streamlit deployment

### Partially Completed / Limited Components

* Hopsworks Feature Store data upload
* Direct Feature Store-based training
* TensorFlow production model
* Separate Flask/FastAPI backend

The project is therefore presented as a working end-to-end machine learning application with the remaining cloud and backend limitations clearly documented.
