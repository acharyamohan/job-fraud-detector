# 🛡️ Job Fraud Detection System

An AI-powered system to detect fraudulent job postings and protect job seekers from scams.

## 🚀 Live Demo

**Try it now:** [Job Fraud Detector](https://job-fraud-detector.streamlit.app/)


## Project Overview

**Job Fraud Detector** is an AI-powered Streamlit web application designed to help users identify potentially fraudulent job postings. The app empowers job seekers, recruiters, and analysts to upload job datasets, analyze postings using both rule-based heuristics and machine learning, and visualize results via interactive dashboards. With features for model training and real-time prediction on new data, this tool aims to increase safety and transparency in the online job market.

---

## Key Features & Technologies Used

### 🔍 Key Features
- **Rule-Based Fraud Detection**  
  Uses heuristic scoring and keyword analysis to flag suspicious job postings.

- **Machine Learning Model Training**  
  Trains a Random Forest classifier using rule-based output as labels to capture complex fraud patterns.

- **Prediction on New Data**  
  Apply the trained model to new or test datasets to classify jobs as fraudulent or genuine.

- **Interactive Dashboard**  
  Visualizes results with pie charts, bar graphs, distribution plots, metrics, and detailed tables.

- **Sample Data & Easy Upload**  
  Test the app instantly with built-in sample data or upload your own CSV files.

- **Downloadable Reports & Models**  
  Download analysis results, trained ML models, and scalers for further use.

### 🛠 Technologies Used
- **Python 3.x**
- **[Streamlit](https://streamlit.io/):** For web app UI and dashboarding
- **[scikit-learn](https://scikit-learn.org/):** Machine learning (Random Forest, scaling, splitting)
- **[pandas](https://pandas.pydata.org/):** Data manipulation and analysis
- **[joblib](https://joblib.readthedocs.io/):** Model and scaler serialization
- **[plotly](https://plotly.com/):** Advanced and interactive charts/visualizations
- **[numpy](https://numpy.org/):** Numerical processing

---

## Setup Instructions

### 1. **Clone the Repository**
```bash
git clone https://github.com/acharyamohan/job-fraud-detector.git
cd job-fraud-detector
