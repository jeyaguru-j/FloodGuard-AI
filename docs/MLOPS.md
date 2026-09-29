\# FloodGuard AI - MLOps Architecture



\## 1. Overview



FloodGuard AI follows an end-to-end MLOps architecture for developing,

deploying and monitoring an urban flood prediction model.



The system is designed around the following lifecycle:



Data Collection → Data Validation → Feature Engineering →

Model Training → Model Evaluation → Model Deployment →

Prediction Monitoring → Drift Detection → Retraining



\---



\## 2. Data Pipeline



Historical weather and flood-event data are collected and transformed

into machine-learning features.



\### Inputs



\- Rainfall

\- Temperature

\- Relative humidity

\- Wind speed

\- Historical flood events

\- Calendar and seasonal information



\### Feature Engineering



The system derives:



\- Rainfall lags

\- Rolling rainfall sums

\- Rolling rainfall maxima

\- Heavy-rain indicators

\- Rainy-day counts

\- Consecutive rainy days

\- Rainfall ratios

\- Temperature statistics

\- Humidity statistics

\- Wind statistics

\- Seasonal indicators



\---



\## 3. Machine Learning Pipeline



The current production prototype uses a Random Forest classifier.



Forecast target:



> Whether at least one flood day occurs during the following

> three days based on information available today.



The trained model contains 52 input features.



\---



\## 4. Model Evaluation



Chronological splitting is used instead of random splitting to reduce

temporal leakage.



\### Test performance



ROC-AUC: 0.8915



PR-AUC: 0.2520



At threshold 0.50:



Precision: 0.2778



Recall: 0.2083



F1-score: 0.2381



Test set size: 730 observations



Positive cases: 24



The dataset is highly imbalanced, therefore ROC-AUC and PR-AUC are

reported in addition to precision, recall and F1-score.



\---



\## 5. Model Serving



The trained model is served through FastAPI.



\### Endpoints



GET `/`



Returns system information.



GET `/health`



Returns API and model health.



GET `/model-info`



Returns model metadata and feature information.



POST `/predict`



Generates a three-day flood-risk prediction.



\---



\## 6. Dashboard



A Streamlit dashboard consumes the FastAPI prediction service.



The dashboard displays:



\- Flood probability

\- Risk level

\- Forecast horizon

\- Rainfall

\- Temperature

\- Humidity

\- Wind speed

\- Key prediction indicators



\---



\## 7. Monitoring



The intended monitoring layer tracks:



\### Data quality



\- Missing values

\- Invalid ranges

\- Schema changes

\- Unexpected data distributions



\### Model performance



\- Precision

\- Recall

\- F1-score

\- ROC-AUC

\- PR-AUC



\### Data drift



Feature distributions are compared between historical

training data and newly arriving data.



Potential tools:



\- Evidently

\- Prometheus

\- Grafana



\---



\## 8. Retraining Strategy



Retraining can be triggered when:



1\. Significant feature drift is detected.

2\. Model performance falls below an acceptable level.

3\. New flood-event data becomes available.

4\. A scheduled retraining interval is reached.



A future automated pipeline can use:



Airflow → Data Ingestion → Validation → Feature Engineering →

Training → Evaluation → Model Registry → Deployment



\---



\## 9. Deployment



The FastAPI service can be containerized using Docker.



The architecture is designed to support future deployment using:



\- Docker

\- Kubernetes

\- Cloud infrastructure

\- CI/CD pipelines



\---



\## 10. Limitations



The historical dataset contains relatively few flood events compared

with non-flood observations.



Therefore:



\- The prediction problem is highly imbalanced.

\- Threshold selection requires further time-series validation.

\- Current API demonstration uses precomputed historical features.

\- Live weather ingestion is a future enhancement.

\- Production deployment would require continuous monitoring and

&#x20; retraining.



These limitations are explicitly documented rather than hidden.

