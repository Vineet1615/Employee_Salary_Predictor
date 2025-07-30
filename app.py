import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    div.block-container {
        padding-top: 1rem;
    }
    [data-testid="stHeader"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Helper Classes & Functions ------------------------

class SalaryPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.usd_to_inr = 83.12
        self.is_trained = False

    @st.cache_data
    def load_and_preprocess_data(_self, uploaded_file):
        """Loads and preprocesses the data from uploaded file or default file."""
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("Salary-Data.csv")
        
        # Drop rows with missing values
        df = df.dropna()
        # Convert to INR
        df['Salary_INR'] = df['Salary'] * _self.usd_to_inr
        return df

    def feature_engineering(self, df):
        df_processed = df.copy()
        categorical_columns = ['Gender', 'Education Level', 'Job Title']
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        feature_columns = [
            'Age', 'Years of Experience',
            'Gender_encoded', 'Education Level_encoded', 'Job Title_encoded'
        ]
        X = df_processed[feature_columns]
        y = df_processed['Salary_INR']
        self.feature_names = feature_columns
        return X, y, df_processed

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        self.is_trained = True
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics

    def predict_salary(self, age, gender, education, job_title, experience):
        """Predict salary for user input"""
        if not self.is_trained:
            return None
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [job_title],
            'Years of Experience': [experience]
        })
        for col in ['Gender', 'Education Level', 'Job Title']:
            if col in self.label_encoders:
                try:
                    input_data[f'{col}_encoded'] = self.label_encoders[col].transform(input_data[col])
                except ValueError:
                    input_data[f'{col}_encoded'] = 0
            else:
                input_data[f'{col}_encoded'] = 0
        X_input = input_data[self.feature_names]
        X_input_scaled = self.scaler.transform(X_input)
        prediction = self.model.predict(X_input_scaled)[0]
        return prediction

# --------------------------- Page Layout/Interface --------------------------

if 'predictor' not in st.session_state:
    st.session_state.predictor = SalaryPredictor()

predictor = st.session_state.predictor

st.markdown("""
    <h2 style="color:#003366;font-family:sans-serif;border-bottom:3px solid #0099CA;padding-bottom:.3em;">
        💰 Employee Salary Predictor
    </h2>
    <span style="color:#555;font-size:1.08em">
    Predict employee salaries in <b>Indian Rupees (INR)</b> based on Age, Gender, Education Level, Job Title, and Experience using advanced Machine Learning.
    </span>
""", unsafe_allow_html=True)
st.divider()

tabs = st.tabs(["🏠 Home", "📈 Data Analysis", "🔮 Salary Prediction", "📋 Model Performance"])
# ------------- Home Tab -----------------
with tabs[0]:
    st.write("""
    ### 🌟 Welcome!
    This app uses **Linear Regression** to help HR teams, candidates, and managers estimate fair salaries.
    
    **How to use:**
    1. [Optional] Upload your own CSV dataset, else the default dataset will be used.
    2. Explore analyses and reports under 'Data Analysis.'
    3. Use 'Salary Prediction' for custom salary estimate.
    4. Check 'Model Performance' for accuracy stats and diagnostics.
    """)
    st.info("The web app will use **Salary-Data.csv** in the app folder if you don't upload your own data. All results are shown in INR ₹.", icon="ℹ️")
    uploaded_file = st.file_uploader("Upload Salary Data (CSV)", type=["csv"], key='uploader')
    st.session_state.df = predictor.load_and_preprocess_data(uploaded_file)

# ------------- Data Analysis Tab -----------------
with tabs[1]:
    st.header("📈 Data Analysis")
    df = st.session_state.df
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.markdown("###### Dataset info:")
    st.write(f"**Rows:** {df.shape[0]} &nbsp; **Columns:** {df.shape[1]}")
    st.divider()

    st.subheader("Salary Distribution (INR)")
    fig1 = px.histogram(df, x='Salary_INR', nbins=30, title='Salary (INR)')
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Salary by Education Level")
        edu_salary = df.groupby('Education Level')['Salary_INR'].mean().reset_index()
        fig2 = px.bar(edu_salary, x='Education Level', y='Salary_INR', 
                      color='Education Level', title="Salary vs Education Level", text_auto='.2s')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Average Salary by Gender")
        gender_salary = df.groupby('Gender')['Salary_INR'].mean().reset_index()
        fig3 = px.bar(gender_salary, x='Gender', y='Salary_INR', 
                      color='Gender', title="Salary vs Gender", text_auto='.2s')
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age vs Salary Scatter")
        st.plotly_chart(px.scatter(df, x='Age', y='Salary_INR', 
                                   color='Gender', symbol='Education Level', title="Age vs Salary"), use_container_width=True)
    with col4:
        st.subheader("Experience vs Salary Scatter")
        st.plotly_chart(px.scatter(df, x='Years of Experience', y='Salary_INR', 
                                   color='Education Level', symbol='Gender', title="Experience vs Salary"), use_container_width=True)

# ------------- Salary Prediction Tab -----------------
with tabs[2]:
    st.header("🔮 Salary Prediction")
    with st.form(key="predict_form", clear_on_submit=False):
        cols = st.columns(2)
        age = cols[0].number_input("Age", min_value=18, max_value=65, value=28, step=1)
        experience = cols[1].number_input("Years of Experience", min_value=0.0, max_value=40.0, value=2.0, step=0.5)
        gender = cols[0].selectbox("Gender", sorted(df['Gender'].unique()))
        education = cols[1].selectbox("Education Level", sorted(df['Education Level'].unique()))
        job_title = st.selectbox("Job Title", sorted(df['Job Title'].unique()))
        submit = st.form_submit_button("Predict Salary 💸")
    
    X, y, _ = predictor.feature_engineering(df)
    predictor.train_model(X, y)

    if submit:
        pred_inr = predictor.predict_salary(age, gender, education, job_title, experience)
        if pred_inr:
            st.success(f"**Estimated Salary: ₹ {pred_inr:,.0f} INR**", icon="💸")
            st.caption("Prediction is based on current patterns in the dataset, using Linear Regression.")

# ------------- Model Performance Tab -----------------
with tabs[3]:
    st.header("📋 Model Performance Metrics")
    X, y, _ = predictor.feature_engineering(df)
    X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics = predictor.train_model(X, y)

    st.markdown(f"""
    <div style="background-color:#e3f2fd;padding:1em 2em;border-radius:10px;border:1px #90caf9 solid">
    <h4 style="color:#195b89;">R² Score</h4>
    <ul>
      <li><b>Training:</b> {metrics['train_r2']:.3f}</li>
      <li><b>Testing:</b> {metrics['test_r2']:.3f}</li>
    </ul>
    <h4 style="color:#195b89;">Root Mean Squared Error (RMSE)</h4>
    <ul>
      <li><b>Training:</b> ₹{metrics['train_rmse']:,.0f}</li>
      <li><b>Testing:</b> ₹{metrics['test_rmse']:,.0f}</li>
    </ul>
    <h4 style="color:#195b89;">Mean Absolute Error (MAE)</h4>
    <ul>
      <li><b>Training:</b> ₹{metrics['train_mae']:,.0f}</li>
      <li><b>Testing:</b> ₹{metrics['test_mae']:,.0f}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("Prediction vs. Actual (Test Data)")
    perf_df = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
    fig = px.scatter(perf_df, x="Actual", y="Predicted", 
                     trendline="ols", title="Actual vs Predicted Salary (Test Set)",
                     labels={'Actual': "Actual Salary (INR)", 'Predicted': "Predicted Salary (INR)"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
