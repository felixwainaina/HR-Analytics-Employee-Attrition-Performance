# Import necessary libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

# Title of the application
st.title("Employee Attrition Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your employee data CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display basic information about the dataset
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Total records: {df.shape[0]}")
    st.write(f"Total columns: {df.shape[1]}")

    # Display column names for user reference
    st.subheader("Column Names")
    st.write(df.columns.tolist())

    # Data Cleaning (if necessary)
    st.subheader("Data Cleaning")
    if st.checkbox("Show missing values"):
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

    # Basic Statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Visualizations
    st.subheader("Attrition Count")
    attrition_count = df['Attrition'].value_counts()
   # fig, ax = plt.subplots()
   # sns.barplot(x=attrition_count.index, y=attrition_count.values, ax=ax)
   # ax.set_title('Attrition Count')
   # ax.set_xlabel('Attrition')
  #  ax.set_ylabel('Count')
  # # st.pyplot(fig)

    # Job Satisfaction vs Attrition
    st.subheader("Job Satisfaction vs Attrition")
   # fig, ax = plt.subplots()
   # sns.boxplot(x='Attrition', y='JobSatisfaction', data=df, ax=ax)
    ax.set_title('Job Satisfaction vs Attrition')
   # st.pyplot(fig)

    # Age Distribution by Attrition
    st.subheader("Age Distribution by Attrition")
   # fig, ax = plt.subplots()
    sns.histplot(data=df, x='Age', hue='Attrition', multiple="stack", bins=30, ax=ax)
    ax.set_title('Age Distribution by Attrition')
   # st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

else:
    st.warning("Please upload a CSV file to proceed.")

# Footer
st.sidebar.header("About")
st.sidebar.text("This application analyzes employee attrition using various metrics.")
