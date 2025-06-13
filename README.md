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
git clone https://github.com/acharyamohan/job-fraud-detector2.git
cd job-fraud-detector2
```

### 2. **Install Required Dependencies**
It’s recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```bash
pip install streamlit pandas scikit-learn joblib plotly numpy
```

### 3. **Run the Application**
```bash
streamlit run app.py
```

### 4. **Using the App**

- **Choose a Mode:**  
  Select from the sidebar:
  - Rule-based Fraud Detection (with Dashboard)
  - Train ML Model Using Rule-based Results
  - Predict on Test Data with Trained ML Model (with Dashboard)

- **Load Data:**  
  Upload your own CSV file or use the provided sample data.  
  CSV columns expected: `title`, `description`, `company`, `location`, `requirements`

- **Explore Results:**  
  View dashboards, download results, and inspect high-risk jobs.

- **Train and Apply ML Model:**  
  Use the detected results to train a model, then apply it to new data.

---

## Example

![Job Fraud Detector Dashboard Screenshot](https://user-images.githubusercontent.com/your_screenshot.png) 



## Acknowledgements

- Inspired by the need for safer online job searching.
- Built with Streamlit, scikit-learn, pandas, and plotly.

---

**Contributions, bug reports, and ideas welcome!**

---

Let me know if you’d like this tailored for a specific deployment environment or if you want a sample screenshot link!
