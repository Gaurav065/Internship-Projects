import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the data
# df = pd.read_csv('loan.csv')
# df.drop('Loan_ID',axis=1)
# # Preprocess the data
# df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
# df['Married'].fillna(df['Married'].mode()[0], inplace=True)
# df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
# df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
# df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
# df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
# df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# # Encode categorical features
# df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
# df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
# df['Education'] = df['Education'].map({'Not Graduate': 0, 'Graduate': 1})
# df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1})
# df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})

# # Split the data into training and testing sets
# X = df.drop('Loan_Status', axis=1)
# y = df['Loan_Status']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
model = joblib.load('RF.C5')

# Define a function to make predictions
def predict_loan_status(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,Total_income):
    # Encode categorical features
    Gender = 0 if Gender == "Male" else 1
    Married = 1 if Married == "Married" else 0
    Education = 1 if Education == "Graduate" else 0
    Self_Employed = 1 if Self_Employed == "Yes" else 0
    Property_Area = 0 if Property_Area == "Rural" else 1 if Property_Area == "Semiurban" else 2
    
    data = {'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area,
            'Total_income':Total_income}
    features = pd.DataFrame(data, index=[0])
    return model.predict(features)[0]

    # Set the app title
st.title("Loan Approval Prediction")
# Add input fields for user data
gender = st.selectbox("Gender", options=["Male", "Female"])
married = st.selectbox("Marital Status", options=["Unmarried", "Married"])
dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 3])
education = st.selectbox("Education", options=["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
income = st.slider("Applicant Income", min_value=150, max_value=81000, step=100)
co_income = st.slider("Co-Applicant Income", min_value=0, max_value=41667, step=100)
loan_amount = st.slider("Loan Amount", min_value=9, max_value=700, step=1)
loan_term = st.selectbox("Loan Term (in years)", options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", options=[0, 1])
property_area = st.selectbox("Property Area", options=["Rural", "Semiurban", "Urban"])
total_income = st.slider("Total Income", min_value=150, max_value=81000, step=100)
# When the user clicks the 'Predict' button, make a prediction
if st.button("Predict"):
    result = predict_loan_status(gender, married, dependents, education, self_employed, income, co_income, loan_amount, loan_term, credit_history, property_area,total_income)
    if result == 0:
        st.error("The loan application was denied.")
    else:
        st.success("The loan application was approved.")
