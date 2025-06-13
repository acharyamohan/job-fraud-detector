import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go

# ========== Rule-based Detection Class ==========
class JobFraudDetector:
    def __init__(self):
        pass

    def extract_features(self, df):
        features = pd.DataFrame()
        features['title_length'] = df['title'].fillna('').str.len()
        features['description_length'] = df['description'].fillna('').str.len()
        features['requirements_length'] = df.get('requirements', pd.Series(['']*len(df))).fillna('').str.len()
        fraud_keywords = [
            'easy money', 'work from home', 'no experience', 'guaranteed income',
            'urgent hiring', 'immediate start', 'cash payment', 'wire transfer',
            'western union', 'money order', 'advance fee', 'processing fee',
            'registration fee', 'training fee', 'equipment fee', 'lottery',
            'inheritance', 'confidential', 'prince', 'beneficiary'
        ]
        text_cols = ['title', 'description', 'requirements']
        for col in text_cols:
            if col in df.columns:
                text_data = df[col].fillna('').str.lower()
                features[f'{col}_fraud_keywords'] = text_data.apply(
                    lambda x: sum(1 for keyword in fraud_keywords if keyword in x)
                )
        features['has_urgency'] = df['description'].fillna('').str.contains(
            r'urgent|immediate|asap|right away|now', case=False, regex=True
        ).astype(int)
        features['has_money_mention'] = df['description'].fillna('').str.contains(
            r'\$\d+|\d+\s*dollars|money|payment|salary|income', case=False, regex=True
        ).astype(int)
        features['has_contact_info'] = df['description'].fillna('').str.contains(
            r'contact|call|email|phone|whatsapp', case=False, regex=True
        ).astype(int)
        features['company_missing'] = (df.get('company', pd.Series(['']*len(df))) == '').astype(int)
        features['company_confidential'] = df.get('company', pd.Series(['']*len(df))).str.contains(
            'confidential|private|undisclosed', case=False, na=False
        ).astype(int)
        features['location_remote'] = df.get('location', pd.Series(['']*len(df))).str.contains(
            'remote|work from home|anywhere', case=False, na=False
        ).astype(int)
        features = features.fillna(0)
        return features

    def detect_fraud_simple(self, job_data):
        fraud_keywords = [
            'easy money', 'work from home', 'no experience needed', 'make money fast',
            'guaranteed income', 'urgent hiring', 'immediate start', 'cash payment',
            'wire transfer', 'western union', 'money order', 'advance fee',
            'lottery', 'inheritance', 'prince', 'beneficiary', 'confidential',
            'processing fee', 'registration fee', 'training fee', 'equipment fee'
        ]

        suspicious_patterns = [
            r'\$\d+\s*(per|\/)\s*(hour|day|week)',  # Unrealistic pay rates
            r'contact.*immediately',
            r'send.*money',
            r'personal.*information',
            r'bank.*details',
            r'social.*security'
        ]

        results_df = job_data.copy()
        results_df['fraud_probability'] = 0.0
        results_df['prediction'] = 'Genuine'
        results_df['risk_level'] = 'Low'

        for idx, job in job_data.iterrows():
            fraud_score = 0
            description = str(job.get('description', '')).lower()
            title = str(job.get('title', '')).lower()
            company = str(job.get('company', '')).lower()
            location = str(job.get('location', '')).lower()
            requirements = str(job.get('requirements', '')).lower()

            all_text = f"{title} {description} {company} {requirements}".lower()
            for keyword in fraud_keywords:
                if keyword in all_text:
                    fraud_score += 0.3

            for pattern in suspicious_patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    fraud_score += 0.25

            if len(description) < 50:
                fraud_score += 0.2
            if '!!!' in title or '$$$' in title:
                fraud_score += 0.4
            if company in ['', 'n/a', 'confidential']:
                fraud_score += 0.3
            if location == 'remote' and fraud_score > 0.3:
                fraud_score += 0.2
            if not requirements or len(requirements) < 20:
                fraud_score += 0.15

            fraud_probability = min(max(fraud_score, 0), 1)
            is_fraud = fraud_probability > 0.5

            results_df.loc[idx, 'fraud_probability'] = round(fraud_probability, 2)
            results_df.loc[idx, 'prediction'] = 'Fraudulent' if is_fraud else 'Genuine'
            results_df.loc[idx, 'risk_level'] = 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.4 else 'Low'

        return results_df

# ========== Streamlit UI ==========
st.set_page_config(page_title="Job Fraud Detection System", page_icon="🛡️", layout="wide")

st.markdown("""
    <div style='text-align:center; margin-bottom:2rem'>
        <h1>🛡️ Job Fraud Detection System</h1>
        <p>Protecting job seekers from fraudulent postings using AI</p>
    </div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Choose what to do:",
    [
        "1. Rule-based Fraud Detection (with Dashboard)",
        "2. Train ML Model Using Rule-based Results",
        "3. Predict on Test Data with Trained ML Model (with Dashboard)"
    ]
)

def get_data_input(section="main"):
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload CSV File", "Use Sample Data"],
        key=f"{section}_input"
    )

    if input_method == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your job dataset (CSV)",
            type=['csv'],
            key=f"{section}_upload"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['title', 'description']
                optional_cols = ['company', 'location', 'requirements']
                missing_required = [col for col in required_cols if col not in df.columns]
                if missing_required:
                    st.error(f"Missing required columns: {missing_required}")
                    return None
                for col in optional_cols:
                    if col not in df.columns:
                        df[col] = ''
                st.success(f"✅ Loaded {len(df)} job postings")
                return df
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        else:
            return None
    else:
        if st.sidebar.button("🎯 Load Sample Data", key=f"{section}_sample"):
            sample_jobs = [
                {
                    'title': 'Software Engineer',
                    'company': 'Tech Corp',
                    'location': 'San Francisco, CA',
                    'description': 'Join our team to build scalable web applications using React and Node.js. 3+ years experience required.',
                    'requirements': "Bachelor's degree in Computer Science, 3+ years React experience, strong problem-solving skills"
                },
                {
                    'title': 'EASY MONEY!!! Work from home NOW!!!',
                    'company': 'Confidential',
                    'location': 'Remote',
                    'description': 'Make $5000 per week working from home! No experience needed! Send $100 registration fee to get started immediately!',
                    'requirements': 'None! Just send money!'
                },
                {
                    'title': 'Data Analyst',
                    'company': 'Analytics Inc',
                    'location': 'New York, NY',
                    'description': 'Analyze large datasets to drive business insights. Python, SQL, and statistical analysis experience required.',
                    'requirements': "Master's degree preferred, 2+ years experience with Python/R, strong analytical skills"
                },
                {
                    'title': 'URGENT! Money Processing Agent',
                    'company': 'Global Finance Solutions',
                    'location': 'Remote',
                    'description': 'Process wire transfers and money orders from home. Guaranteed $3000 weekly income! Contact immediately!',
                    'requirements': 'Must have bank account for processing payments'
                },
                {
                    'title': 'Marketing Manager',
                    'company': 'Brand Solutions LLC',
                    'location': 'Chicago, IL',
                    'description': 'Lead marketing campaigns for B2B clients. Develop strategies, manage budgets, and analyze performance metrics.',
                    'requirements': '5+ years marketing experience, MBA preferred, strong communication skills'
                },
                {
                    'title': 'Customer Service Representative',
                    'company': 'ServicePlus Inc',
                    'location': 'Austin, TX',
                    'description': 'Handle customer inquiries and provide excellent service. Full training provided.',
                    'requirements': 'High school diploma, good communication skills, customer service experience preferred'
                }
            ]
            st.session_state[f'{section}_sample_loaded'] = True
            st.session_state[f'{section}_sample_df'] = pd.DataFrame(sample_jobs)
            return st.session_state[f'{section}_sample_df']
        if st.session_state.get(f'{section}_sample_loaded'):
            return st.session_state[f'{section}_sample_df']
        return None

detector = JobFraudDetector()

def dashboard(df_results):
    total_jobs = len(df_results)
    fraudulent_jobs = len(df_results[df_results['prediction'] == 'Fraudulent'])
    genuine_jobs = len(df_results[df_results['prediction'] == 'Genuine'])
    high_risk_jobs = len(df_results[df_results['risk_level'] == 'High'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", total_jobs)
    with col2:
        st.metric("Fraudulent", fraudulent_jobs)
    with col3:
        st.metric("Genuine", genuine_jobs)
    with col4:
        st.metric("High Risk", high_risk_jobs)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Results Table", "⚠️ Suspicious Jobs", "📈 Analytics"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                values=[genuine_jobs, fraudulent_jobs],
                names=['Genuine', 'Fraudulent'],
                title="Job Classification Distribution",
                color_discrete_map={'Genuine': '#10B981', 'Fraudulent': '#EF4444'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            risk_counts = df_results['risk_level'].value_counts()
            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
            )
            fig_risk.update_layout(showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)

    with tab2:
        st.subheader("📋 All Job Analysis Results")
        st.dataframe(df_results, use_container_width=True, height=400)
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("⚠️ Most Suspicious Job Postings")
        suspicious_jobs = df_results[df_results['prediction'] == 'Fraudulent'].sort_values(
            'fraud_probability', ascending=False
        )
        if len(suspicious_jobs) > 0:
            for idx, job in suspicious_jobs.iterrows():
                with st.expander(f"🚨 {job['title']} - Risk Score: {job['fraud_probability']:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Company:** {job.get('company', 'N/A')}")
                        st.write(f"**Location:** {job.get('location', 'N/A')}")
                        st.write(f"**Risk Level:** {job['risk_level']}")
                    with col2:
                        st.write(f"**Fraud Probability:** {job['fraud_probability']:.2f}")
                        st.write(f"**Classification:** {job['prediction']}")
                    st.write("**Description:**")
                    st.write(job.get('description', 'N/A'))
                    if job.get('requirements'):
                        st.write("**Requirements:**")
                        st.write(job['requirements'])
        else:
            st.success("🎉 No suspicious jobs found in this dataset!")

    with tab4:
        st.subheader("📈 Fraud Detection Analytics")
        fig_hist = px.histogram(
            df_results,
            x='fraud_probability',
            nbins=20,
            title="Fraud Probability Distribution",
            labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Jobs'}
        )
        fig_hist.update_traces(marker_color='lightblue', marker_line_color='navy', marker_line_width=1)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📊 Key Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Common Fraud Indicators Found:**")
            fraud_jobs = df_results[df_results['prediction'] == 'Fraudulent']
            if len(fraud_jobs) > 0:
                fraud_keywords = ['urgent', 'immediate', 'easy money', 'guaranteed', 'no experience']
                found_keywords = []
                for keyword in fraud_keywords:
                    count = fraud_jobs['description'].str.contains(keyword, case=False, na=False).sum()
                    if count > 0:
                        found_keywords.append(f"• '{keyword}': {count} jobs")
                if found_keywords:
                    for keyword in found_keywords:
                        st.write(keyword)
                else:
                    st.write("• No common fraud keywords detected in this sample")
        with col2:
            st.success("**Protection Tips:**")
            st.write("• Be wary of jobs requiring upfront payments")
            st.write("• Verify company information independently")
            st.write("• Avoid jobs with unrealistic salary promises")
            st.write("• Check for proper contact information")
            st.write("• Research the company's online presence")

if mode == "1. Rule-based Fraud Detection (with Dashboard)":
    df = get_data_input("main")
    if df is not None:
        df_results = detector.detect_fraud_simple(df)
        dashboard(df_results)
    else:
        st.info("Please upload a dataset or load sample data.")

elif mode == "2. Train ML Model Using Rule-based Results":
    df = get_data_input("main")
    if df is not None:
        df_results = detector.detect_fraud_simple(df)
        features = detector.extract_features(df_results)
        labels = (df_results['prediction'] == 'Fraudulent').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=False)
        st.text("Classification Report (on 20% validation split):")
        st.text(report)
        joblib.dump(model, 'fraud_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.success("ML model and scaler trained and saved as 'fraud_model.pkl' and 'scaler.pkl'.")
        with open('fraud_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='fraud_model.pkl')
        with open('scaler.pkl', 'rb') as f:
            st.download_button('Download Scaler', f, file_name='scaler.pkl')
    else:
        st.info("Please upload a dataset or load sample data.")

elif mode == "3. Predict on Test Data with Trained ML Model (with Dashboard)":
    df = get_data_input("test")
    if not (os.path.exists('fraud_model.pkl') and os.path.exists('scaler.pkl')):
        st.warning("Please train the ML model first (use mode 2)!")
    elif df is not None:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        test_features = detector.extract_features(df)
        test_features_scaled = scaler.transform(test_features)
        probabilities = model.predict_proba(test_features_scaled)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        df_results = df.copy()
        df_results['fraud_probability'] = probabilities
        df_results['prediction'] = ['Fraudulent' if p else 'Genuine' for p in predictions]
        df_results['risk_level'] = [
            'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
            for prob in probabilities
        ]
        dashboard(df_results)
    else:
        st.info("Please upload test data or load sample test data.")

else:
    st.info("Please choose a mode to begin.")
