import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set up the title of the app
st.title("Employee Attrition Analysis")

# Step 1: Upload CSV file
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.write("Data Loaded Successfully!")
    
    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.write(data.head())
    
    # Check for missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    # Data Cleaning: Drop missing values for simplicity
    data.dropna(inplace=True)

    # Encode categorical variables using Label Encoding
    categorical_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 
                           'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    
    le = LabelEncoder()
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])
    
    # Step 2: Exploratory Data Analysis (EDA)
    
    # Attrition Rate Calculation
    attrition_rate = data['Attrition'].value_counts(normalize=True)[1] * 100
    st.subheader(f'Attrition Rate: {attrition_rate:.2f}%')

    # Correlation Analysis
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot(plt)

    # Visualizations: Age Distribution by Attrition Status
    st.subheader("Age Distribution by Attrition Status")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Age', hue='Attrition', multiple='stack', bins=30)
    plt.title('Age Distribution by Attrition Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Monthly Income by Attrition Status
    st.subheader("Monthly Income by Attrition Status")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)
    plt.title('Monthly Income by Attrition Status')
    plt.xlabel('Attrition')
    plt.ylabel('Monthly Income')
    st.pyplot(plt)

    # Job Role vs. Attrition
    st.subheader("Job Role vs. Attrition")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='JobRole', hue='Attrition', data=data)
    plt.title('Job Role vs. Attrition')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Step 3: Predictive Modeling
    
    # Define features and target variable
    X = data.drop(['Attrition'], axis=1)
    y = data['Attrition']

    # Split the dataset into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader(f'Accuracy of Logistic Regression Model: {accuracy:.2f}')
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    st.subheader('Confusion Matrix')
    
    fig, ax = plt.subplots()
    
   sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
   ax.set_xlabel('Predicted')
   ax.set_ylabel('Actual')
   ax.set_title('Confusion Matrix')
   st.pyplot(fig)

