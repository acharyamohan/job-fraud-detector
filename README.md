
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
  Uses heuristic scoring and keyword analysis to flag suspicious job postings.

  **Checks for:**
  - Fraudulent keywords in title/description/requirements (e.g., "easy money", "urgent hiring").
  - Suspicious patterns (e.g., unrealistic pay, requests for money, missing company info).
  - Short or generic descriptions, excessive punctuation.
  - Remote/work-from-home indicators.

  **Outputs:**
  - Fraud probability (0-1)
  - Prediction label (“Fraudulent”/“Genuine”)
  - Risk level (“High”/“Medium”/“Low”)


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

## 🧠 Data Science Workflow

### 1. 📥 Input & Feature Engineering
Users upload job data with required columns. Features are extracted including:
- Title/description lengths
- Fraud keyword (‘easy money’, ‘urgent hiring’, or ‘no experience needed’) counts
- Flags for suspicious patterns (e.g., urgency, remote jobs)

```python
features = detector.extract_features(df)
```

### 2. 🧾 Rule-Based Detection (Mode 1)
Simple heuristics assign fraud probability:

- Keyword and regex matching
- Score threshold:  
  - Fraudulent if score > 0.5  
  - High risk: > 0.7, Medium: 0.4–0.7, Low: < 0.4

### 3. 🤖 Machine Learning (Mode 2 & 3)

#### 🛠 Training
- Uses rule-based output as pseudo labels
- Features scaled via `StandardScaler`
- `RandomForestClassifier` trained on 80/20 split
- Model saved as `fraud_model.pkl`

```python
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

#### 🔍 Prediction
- Loads the model and scaler
- Applies to new test data
- Full dashboard with predicted labels, risk scores, charts

---

## 📁 Resources

- 📦 [Download Model (`fraud_model.pkl`)](https://drive.google.com/...)
- 📦 [Download Scaler (`scaler.pkl`)](https://drive.google.com/...)
- 📄 [Sample Dataset](https://drive.google.com/...)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Please fork the repo and open a pull request.

---
