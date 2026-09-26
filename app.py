import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile

# Configure the Streamlit page layout
st.set_page_config(page_title="Liver Patient Analysis", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_and_clean_data():
    """Extracts and cleans the dataset directly from the zip archive."""
    with zipfile.ZipFile("archive (3).zip", 'r') as zip_ref:
        with zip_ref.open("Indian Liver Patient Dataset (ILPD).csv") as file:
            # Read normally, letting Pandas automatically pick up the headers
            df = pd.read_csv(file)
            
    # Impute missing values for the ag_ratio column using the median
    if 'ag_ratio' in df.columns:
        df['ag_ratio'] = df['ag_ratio'].fillna(df['ag_ratio'].median())
    
    # Create a clear categorical target variable for visualizations
    if 'is_patient' in df.columns:
        df['Diagnosis'] = df['is_patient'].map({1: 'Liver Patient', 2: 'Healthy'})
        
    return df

df = load_and_clean_data()

# --- SIDEBAR INTERACTIVITY ---
st.sidebar.title("Dashboard Controls")
st.sidebar.write("Filter the dataset to update the charts in real-time.")

# Dynamic filters based on dataset limits
min_age, max_age = int(df['age'].min()), int(df['age'].max())
selected_age = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

selected_gender = st.sidebar.multiselect("Select Gender", options=df['gender'].unique(), default=df['gender'].unique())
selected_diagnosis = st.sidebar.multiselect("Select Diagnosis", options=df['Diagnosis'].unique(), default=df['Diagnosis'].unique())

# Apply user filters
filtered_df = df[
    (df['age'] >= selected_age[0]) & 
    (df['age'] <= selected_age[1]) &
    (df['gender'].isin(selected_gender)) &
    (df['Diagnosis'].isin(selected_diagnosis))
]

# Stop execution gracefully if filters remove all data
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your sidebar selections.")
    st.stop()

# --- MAIN PAGE LAYOUT ---
st.title("Indian Liver Patient Dataset Analysis")
st.info ('Note :- This data set contains **416 liver patient records** and 167 non liver patient records.The data set was collected from test samples in **North East of Andhra Pradesh, India**.')

tab_overview, tab_distributions, tab_correlations,depth = st.tabs([
    "Dataset Overview", 
    "Feature Distributions", 
    "Correlation Analysis",
    "In Depth Statistical Analysis"
])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    # Top-level KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(filtered_df))
    col2.metric("Average Age", f"{filtered_df['age'].mean():.1f} yrs")
    col3.metric("Male Patients", len(filtered_df[filtered_df['gender'] == 'Male']))
    col4.metric("Female Patients", len(filtered_df[filtered_df['gender'] == 'Female']))
    
    st.divider()
    
    col_chart, col_stats = st.columns([1, 1.5])
    
    with col_chart:
        st.subheader("Diagnosis Proportion")
        fig_pie = px.pie(
            filtered_df, 
            names='Diagnosis', 
            hole=0.4,
            color='Diagnosis',
            color_discrete_map={'Liver Patient': '#EF553B', 'Healthy': '#00CC96'}
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_stats:
        st.subheader("Statistical Summary")
        # Exclude 'is_patient' integer column from descriptive stats as we use 'Diagnosis'
        stats_df = filtered_df.drop(columns=['is_patient'], errors='ignore').describe()
        st.dataframe(stats_df, use_container_width=True)

# --- TAB 2: DISTRIBUTIONS ---
with tab_distributions:
    st.subheader("Interactive Feature Distributions")
    
    # Identify numeric columns for the dropdown
    numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'is_patient' in numeric_columns:
        numeric_columns.remove('is_patient')
        
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        feature_to_plot = st.selectbox("Select Clinical Feature:", numeric_columns)
    with ctrl_col2:
        plot_type = st.radio("Visualization Style:", ["Histogram", "Box Plot", "Violin Plot"], horizontal=True)
        
    # Render the chosen plot style
    if plot_type == "Histogram":
        fig_dist = px.histogram(
            filtered_df, x=feature_to_plot, color="Diagnosis", 
            barmode="overlay", opacity=0.7, marginal="box"
        )
    elif plot_type == "Box Plot":
        fig_dist = px.box(filtered_df, x="Diagnosis", y=feature_to_plot, color="Diagnosis")
    else:
        fig_dist = px.violin(filtered_df, x="Diagnosis", y=feature_to_plot, color="Diagnosis", box=True)
        
    st.plotly_chart(fig_dist, use_container_width=True)

# --- TAB 3: CORRELATIONS ---
with tab_correlations:
    st.subheader("Clinical Feature Correlations")
    
    # Calculate correlation matrix
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    if 'is_patient' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['is_patient'])
        
    corr_matrix = numeric_df.corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)
with depth :
    # --- IN-DEPTH STATISTICAL ANALYSIS ---
    st.markdown("---")
    st.header("In-Depth Statistical Analysis")
    st.write("Explore detailed clinical relationships, demographic breakdowns, and enzyme trends.")

    # --- 1. Age Demographics (Pie Chart) ---
    st.subheader("1. Patient Age Demographics")
    # Create age categories for better grouping
    filtered_df['Age_Group'] = pd.cut(
        filtered_df['age'], 
        bins=[0, 30, 50, 70, 100], 
        labels=['Under 30', '30-50', '50-70', '70+']
    )
    age_counts = filtered_df['Age_Group'].value_counts().reset_index()
    age_counts.columns = ['Age_Group', 'Count']

    fig_age_pie = px.pie(
        age_counts, 
        names='Age_Group', 
        values='Count', 
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Teal
    )
    st.plotly_chart(fig_age_pie, use_container_width=True)
    st.info("**Statistical Insight:** This donut chart breaks down the patient population into distinct age groups. Liver diseases often manifest at different rates depending on age, making demographic segmentation essential for identifying high-risk categories.")

    # --- 2. Enzyme Levels by Gender & Diagnosis (Bar Graph) ---
    st.subheader("2. Average Enzyme Levels by Demographics")
    # Let user select which enzyme to analyze
    enzyme_options = ['alkphos', 'sgpt', 'sgot', 'tot_bilirubin']
    selected_enzyme = st.selectbox("Select an Enzyme to analyze:", enzyme_options)

    # Calculate means
    avg_enzyme = filtered_df.groupby(['gender', 'Diagnosis'])[selected_enzyme].mean().reset_index()

    fig_bar = px.bar(
        avg_enzyme, 
        x='gender', 
        y=selected_enzyme, 
        color='Diagnosis', 
        barmode='group',
        text_auto='.2f',
        color_discrete_map={'Liver Patient': '#EF553B', 'Healthy': '#00CC96'}
    )
    fig_bar.update_layout(yaxis_title=f"Average {selected_enzyme}")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.info(f"**Statistical Insight:** This grouped bar chart compares the average levels of **{selected_enzyme}** across genders and diagnosis status. It highlights whether elevated clinical metrics are uniform across sexes or if disparities exist. Liver patients typically demonstrate markedly higher averages.")

    # --- 3. Protein Trends Across Ages (Line Chart) ---
    st.subheader("3. Protein Level Trends Over Time")
    # Calculate the mean protein levels for each exact age
    age_trend = filtered_df.groupby('age')[['tot_proteins', 'albumin']].mean().reset_index()

    fig_line = px.line(
        age_trend, 
        x='age', 
        y=['tot_proteins', 'albumin'],
        labels={'value': 'Concentration Level', 'variable': 'Protein Type', 'age': 'Age'}
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.info("**Statistical Insight:** This line chart tracks average Total Proteins and Albumin levels as patients age. Fluctuations or a general downward trend in albumin can serve as an indicator of declining liver function or malnourishment in older populations.")

    # --- 4. Bilirubin Correlation (Scatter Bubble Chart) ---
    st.subheader("4. Total vs. Direct Bilirubin Analysis")
    fig_scatter = px.scatter(
        filtered_df, 
        x='tot_bilirubin', 
        y='direct_bilirubin', 
        color='Diagnosis', 
        size='age', 
        hover_data=['gender'],
        color_discrete_map={'Liver Patient': '#EF553B', 'Healthy': '#00CC96'}
    )
    fig_scatter.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.info("**Statistical Insight:** Total and Direct Bilirubin usually share a strong linear positive correlation. Outliers or heavy clustering separated by diagnosis highlight the severity of liver distress. The size of each bubble represents the patient's age, adding a third dimension to the analysis.")

import streamlit as st

st.divider()
st.subheader("About the Developer :-")

# Custom CSS and HTML for the profile card
custom_profile_card = """
<style>
.profile-card {
    background-color: #1e1e2f;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    color: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border-left: 5px solid #00f2fe;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin-bottom: 20px;
}
.profile-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 20px rgba(0,0,0,0.4);
}
.profile-title {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 5px;
    background: -webkit-linear-gradient(#4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.profile-subtitle {
    font-size: 16px; 
    color: #a9a9b3; 
    margin-top: 0;
}
.github-btn {
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    border: none;
    color: white !important;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 14px;
    border-radius: 25px;
    margin-top: 15px;
    font-weight: bold;
    transition: opacity 0.2s;
}
.github-btn:hover {
    opacity: 0.9;
    text-decoration: none;
}
.credit-text {
    font-size: 14px; 
    color: #a9a9b3;
    line-height: 1.5;
}
</style>

<div class="profile-card">
    <p style="font-size: 12px; color: #00f2fe; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 1.5px;">Personal Details</p>
    <div class="profile-title">YASH GARG</div>
    <p class="profile-subtitle">16 Years | Student | Analyst | Developer</p>
    <hr style="border: 1px solid #333; margin: 15px 0;">
    <p class="credit-text">Special thanks to <b>Kaggle<b> for providing the world-class datasets that make this research possible.</p>
    <a href="https://github.com/Focus-on-Future" target="_blank" class="github-btn">GitHub Profile</a>
</div>
"""

# Render the card in your Streamlit app
st.markdown(custom_profile_card, unsafe_allow_html=True)