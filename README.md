
# 🛡️ Job Fraud Detection System

An AI-powered tool that detects fraudulent job postings to protect job seekers from scams.

## 🚀 Live Demo
👉 [Try the App on Streamlit](https://job-fraud-detector.streamlit.app/)

---

## 📑 Table of Contents

- [Live Demo](#-live-demo)
- [Project Overview](#project-overview)
- [Key Features & Technologies](#-key-features--technologies-used)
- [Setup Instructions](#setup-instructions)
- [Dashboard & Outputs](#dashboard--outputs)
- [Example Use Cases](#example-use-cases)
- [Extensibility](#extensibility)
- [Screenshots](#-screenshots)
- [Data Science Workflow](#data-science-workflow)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📘 Project Overview

**Job Fraud Detector** is a Streamlit-based AI web app that helps users detect potentially fraudulent job postings. Users can analyze datasets using rule-based heuristics and machine learning models, and view results through rich interactive dashboards.

---

## 🔍 Key Features & Technologies Used

### 🔍 Features

- **Rule-Based Fraud Detection**  
  - Scans for fraudulent keywords (e.g., "easy money", "urgent hiring")
  - Flags suspicious job characteristics (e.g., missing company info, unrealistic salary)
  - Evaluates short/generic descriptions, excessive punctuation, etc.
  - Generates fraud **probability**, **label**, and **risk level**

- **Machine Learning Model**  
  - Trains a `RandomForestClassifier` based on rule-based detections
  - Saves and reuses model (`fraud_model.pkl`) and scaler (`scaler.pkl`)

- **Real-Time Prediction**  
  - Apply the trained model to new/test data
  - Visual output with charts and dashboards

- **Interactive Dashboard**  
  - Metrics summary, pie & bar charts, probability histograms
  - Styled results table with fraud level indicators
  - Expandable panels for suspicious job details

- **Easy Upload & Download**  
  - Upload CSV files or use built-in sample data
  - Export results and models for reuse

---

### 🛠 Technologies Used

- **Python 3.x**
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [plotly](https://plotly.com/)
- [joblib](https://joblib.readthedocs.io/)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/acharyamohan/job-fraud-detector.git
cd job-fraud-detector
```

### 2. Install Dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas scikit-learn joblib plotly numpy
```

### 3. Run the App

```bash
streamlit run app.py
```

---

### 4. Expected CSV Format

Your dataset should include the following columns:

| Column        | Description                      | Required |
|---------------|----------------------------------|----------|
| `title`       | Job title                        | ✅       |
| `description` | Job description                  | ✅       |
| `company`     | Company name                     | ❌       |
| `location`    | Job location                     | ❌       |
| `requirements`| Skills or qualifications         | ❌       |

---

## 📊 Dashboard & Outputs

- **Top Metrics**: Total, genuine, fraudulent, and high-risk job counts
- **Pie Chart**: Fraudulent vs. genuine postings
- **Bar Chart**: Risk level breakdown
- **Histogram**: Fraud probability distribution
- **Results Table**: Color-coded predictions
- **Suspicious Jobs**: Expandable sections for high-risk posts
- **Downloadable Files**: Results CSV, model `.pkl`, and scaler `.pkl`

---

## 🧪 Example Use Cases

- **Job Seekers**: Detect scams before applying
- **Recruiters**: Audit platforms for fake or misleading job posts
- **Analysts**: Study patterns and behaviors in fraudulent job markets

---

## 🔄 Extensibility

- Add more advanced NLP models (e.g., BERT, LSTM)
- Integrate actual labeled fraud datasets for supervised learning
- Add authentication and API integration for job boards
- Deploy to cloud platforms (e.g., AWS, GCP, Heroku)

---

## 📸 Screenshots

### Dashboard Overview  
![Dashboard](https://raw.githubusercontent.com/acharyamohan/job-fraud-detector/main/dash_board_overview.png)

### Upload & Results  
![Upload](https://raw.githubusercontent.com/acharyamohan/job-fraud-detector/main/upload_and_results.png)

### Visual Analysis  
![Graphs](https://raw.githubusercontent.com/acharyamohan/job-fraud-detector/main/visual_analysis.png)

### Suspicious Postings  
![Details](https://raw.githubusercontent.com/acharyamohan/job-fraud-detector/main/suspicious_job.png)

---

## 🧠 Data Science Workflow

### 1. 📥 Input & Feature Engineering
Users upload job data with required columns. Features are extracted including:
- Title/description lengths
- Fraud keyword counts
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

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
