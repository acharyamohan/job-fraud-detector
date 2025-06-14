
# 🛡️ Job Fraud Detection System

An AI-powered system to detect fraudulent job postings and protect job seekers from scams.

## 🚀 Live Demo

**Try it now:** [Job Fraud Detector](https://job-fraud-detector.streamlit.app/)

---

## Project Overview

**Job Fraud Detector** is an AI-powered Streamlit web application designed to help users identify potentially fraudulent job postings. The app empowers job seekers, recruiters, and analysts to upload job datasets, analyze postings using both rule-based heuristics and machine learning, and visualize results via interactive dashboards. With features for model training and real-time prediction on new data, this tool aims to increase safety and transparency in the online job market.

---

## Key Features & Technologies Used

### 🔍 Key Features
- **Rule-Based Fraud Detection**  
  Uses heuristic scoring and keyword analysis to flag suspicious job postings
•	Checks for:
o	Fraudulent keywords in title/description/requirements (e.g., "easy money", "urgent hiring").
o	Suspicious patterns (e.g., unrealistic pay, requests for money, missing company info).
o	Short or generic descriptions, excessive punctuation.
o	Remote/work-from-home indicators.
•	Outputs:
o	Fraud probability (0-1)
o	Prediction label (“Fraudulent”/“Genuine”)
o	Risk level (“High”/“Medium”/“Low”)


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

### 5. **Dashboard & Outputs**
- **Metrics row:** Total, fraudulent, genuine, high-risk jobs  
- **Pie chart:** Fraudulent vs. genuine distribution  
- **Bar chart:** Risk levels  
- **Histogram:** Fraud probability scores  
- **Results table:** Styled by fraud/risk  
- **Detailed suspicious jobs:** Expandable panels  
- **Download buttons:** CSV for results, trained model/scaler  

---

### 6. **Example Use Cases**
- **Job Seekers:** Check if a posting is suspicious before applying  
- **Recruiters:** Audit job boards for scam/fake ads  
- **Analysts:** Study trends in fraudulent postings  

---

### 7. **Extensibility**
- Easily add more features (new keywords, ML models, NLP analysis)  
- Integrate with actual labeled data for supervised ML  
- Deploy to cloud for public access  

---

## 📸 Example Screenshots

### Dashboard Overview
![Dashboard](https://github.com/acharyamohan/job-fraud-detector/blob/6dcf52a4f1deb425d428a1c57dcc6b0d49b9e9e4/dash_board_overview.png)

### Upload & Results
![Upload](https://github.com/acharyamohan/job-fraud-detector/blob/6dcf52a4f1deb425d428a1c57dcc6b0d49b9e9e4/upload_and_results.png)

### Visual Analysis
![Graphs](https://github.com/acharyamohan/job-fraud-detector/blob/6dcf52a4f1deb425d428a1c57dcc6b0d49b9e9e4/visual_analysis.png)

### Suspicious Postings
![Details](https://github.com/acharyamohan/job-fraud-detector/blob/6dcf52a4f1deb425d428a1c57dcc6b0d49b9e9e4/suspicious_job.png)

---

## Acknowledgements

- Inspired by the need for safer online job searching.  
- Built with Streamlit, scikit-learn, pandas, and plotly.

---

**Contributions, bug reports, and ideas welcome!**
