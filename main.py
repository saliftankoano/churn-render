import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

from utils import create_gauge_chart, create_model_probability_chart

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ['GROQ_API_KEY']
)
# Model loader function
def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)
# Load all models
gb_model = load_model("gb_model-SMOTE.pkl")
xgboost_model = load_model("xgboost_model-SMOTE.pkl")
random_forest_model = load_model("rf_model-SMOTE.pkl")
voting_classifier_model = load_model("voting_clf_model.pkl")

def prepare_input(CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
  IsActiveMember, EstimatedSalary, location, gender):
  input_dict = {
    "CreditScore": CreditScore,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": int(HasCrCard),
    "IsActiveMember": int(IsActiveMember),
    "EstimatedSalary": EstimatedSalary,
    "Geography_France": 1 if location == "France" else 0,
    "Geography_Germany": 1 if location == "Germany" else 0,
    "Geography_Spain": 1 if location == "Spain" else 0,
    "Gender_Female": gender == "Female",
    "Gender_Male": gender == "Male",
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_prediction(input_df, input_dict):
  probabilities = {
    'Gradient Boosting': gb_model.predict_proba(input_df)[0][1],
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
    'Voting Classifier': voting_classifier_model.predict_proba(input_df)[0][1],
  }
  avg_probability = np.mean(list(probabilities.values()))
    
  col1, col2 = st.columns(2)
  with col1:
    fig = create_gauge_chart(avg_probability)
    st.plotly_chart(fig)
    st.write(f"The customer has a: {avg_probability:.2f} probablity of churning.")
  with col2:
    fig_probs= create_model_probability_chart(probabilities)
    assert isinstance(fig_probs, go.Figure), "fig_probs is not a Plotly Figure"
    st.plotly_chart(fig_probs, use_container_width=True)
  
  return avg_probability

def explain_prediction(probability, input_dict, surname):
  prompt = f"""
  You are an expert data scientist at a bank, specializing in explaining
  customer churn predictions. The system has identified that {surname}
  has a {round(probability*100, 1)}% chance of leaving the bank, based
  on their profile and the factors listed below.

  Customer Profile:
  {input_dict}

  Top 10 Features Influencing Churn:
  Feature | Importance
  NumOfProducts      0.323888
  IsActiveMember     0.164146
  Age                0.109550
  Geography_Germany  0.091373
  Balance            0.052786
  Geography_France   0.046463
  Gender_Female      0.045283
  Geography_Spain    0.036855
  CreditScore        0.035005
  EstimatedSalary    0.032655
  HasCrCard          0.031940
  Tenure             0.030054
  Gender_Male        0.000000

  Here are the summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are the summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

  Based on the customer’s profile and comparison with churned and non-churned customers:

  - If the customer’s risk of churning is over 40%, provide a brief, 3-
  sentence explanation of why they might be at risk of leaving the bank.
  - If the customer’s risk is below 40%, offer a 3-sentence explanation
  of why they are likely to remain a customer.Avoid mentioning
  probabilities, machine learning models, or directly referencing
  technical aspects like feature importance. Focus on providing clear,
  intuitive reasons for churn based on the customer’s information.
  """

  
  print("Explanation prompt: ", prompt)
  raw_response = client.chat.completions.create(
    model= "llama-3.2-3b-preview",
    messages= [{
      "role": "user",
      "content": prompt
    }]
  )
  
  return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""
  You are Jason Duval, a Senior Account Executive at Genos
  Bank. Your role is to ensure customers remain satisfied
  with the bank and to offer personalized incentives to
  strengthen their relationship with us.

  You’ve identified that {surname}, one of our valued
  customers, might benefit from additional support and
  tailored offerings to enhance their banking experience.

  Customer Information:
  {input_dict}

  Explanation of why the customer may be at risk:
  {explanation}

  Based on this, write a warm, reassuring, and persuasive
  email to the customer. The email should emphasize Genos
  Bank’s commitment to supporting their financial needs
  and growth. Offer a personalized set of incentives to
  encourage them to continue banking with us. The tone
  should be positive, focusing on how our bank can serve
  as a trusted partner in achieving their financial goals.

  Include a set of incentive offerings in bullet point
  format, and after each bullet point, ensure a line
  break. Do not mention anything about their probability
  of churning, the machine learning model, or any negative
  aspects of their situation. Instead, position the bank
  as a proactive solution provider.

  Avoid referencing specific numerical values for their
  balance or estimated income. Focus on their overall
  relationship with the bank and how these offerings can
  enhance their experience.
  """

  raw_response = client.chat.completions.create(
    model= "llama-3.2-3b-preview",
    messages= [{
      "role": "user",
      "content": prompt
    }],
  )
  print(" \n\nEmail prompt: ", prompt)
  
  return raw_response.choices[0].message.content
st.title("Genos Bank customer churn prediction")
df = pd.read_csv("churn.csv")

# Customer id and surname
customers = [f"{row['CustomerId']} - {row['Surname']}" for _,row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print("Selected customer ID: ", selected_customer_id)
  selected_customer_surname = selected_customer_option.split(" - ")[1]
  print("Selected customer surname: ", selected_customer_surname)
  # Identify selected customer
  selected_customer = df.loc[df['CustomerId'] 
  ==  selected_customer_id].iloc[0]
  print("Selected customer: ", selected_customer)
  # Setup 2 columns layout 
  col1,col2 = st.columns(2)
  # Assign UI elements to columns

  with col1:
    credit_score = st.number_input(
      "Credit Score",
      min_value= 300,
      max_value= 850,
      value = int(selected_customer['CreditScore'])
    )
    location = st.selectbox(
      "Locaton", ["Spain", "France", "Germany"],
      index= ["Spain", "France", "Germany"].index(selected_customer['Geography'])
    )
    gender = st.radio(
      "Gender",
      ["male", "female"], 
      index= 0 if selected_customer['Gender'] == "Male" else 1
    )
    age = st.number_input(
      "Age",
      min_value= 18,
      max_value= 100,
      value = int(selected_customer['Age'])
    )
    tenure = st.number_input(
      "Tenure (years)",
      min_value= 0,
      max_value= 50,
      value = int(selected_customer['Tenure'])
    )

  with col2:
    balance = st.number_input(
      "Balance",
      min_value= 0.0,
      value= float(selected_customer['Balance'])
    )
    num_products = st.number_input(
      "Number of products",
      min_value= 0,
      value= int(selected_customer['NumOfProducts'])
    )
    has_credit_card = st.checkbox(
      "Has credit card",
      value= bool(selected_customer['HasCrCard'])
    )
    is_active_member = st.checkbox(
      "Is active member",
      value= bool(selected_customer['IsActiveMember'])
    )
    estimated_salary = st.number_input(
      "Estimated salary",
      min_value= 0.0,
      value= float(selected_customer['EstimatedSalary'])
    )
  
  age_ratio_tenure = df['CustomerId'][df['Age']] / df['CustomerId'][df['Tenure']]
  # Make RowNumber and CustomerId columns categorical to be able to use the model
  RowNumber = df['RowNumber'].astype('category')
  customerId = df['CustomerId'].astype('category')
  
  input_df, input_dict = prepare_input(credit_score, age, tenure,
  balance, num_products, has_credit_card, is_active_member,
  estimated_salary, location, gender)
  
  avg_probability = make_prediction(input_df, input_dict)
  explanation = explain_prediction(avg_probability, input_dict,
  selected_customer_surname)
  email = generate_email(avg_probability, input_dict,
  explanation,selected_customer['Surname'])
  
  # Formating explanation
  st.markdown("------")
  st.subheader("Explanation of the prediction: ")
  st.markdown(explanation)
  
  # Generate email
  st.markdown("------")
  st.subheader("Personalize customer email: ")
  st.markdown(email)
else:
  selected_customer_id = None


