import os
import pickle
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from openai import OpenAI

from utils import (
  create_fraud_gauge_chart,
  create_gauge_chart,
  create_model_probability_chart,
)

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
# Function to prepare the churn prediciton data
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
# Function to predict churn probability
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
# Function to predict fraud probability
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
# Function to predict churn probability
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

# Load fraud detection model
fraud_detection_model = load_model("fraud_model.pkl")
# Prepare input for fraud detection model
def prepare_fraud_input(selected_customer):
  customer_data = pd.DataFrame([selected_customer])
  customer_data['state_code'] = customer_data['state'].astype('category').cat.codes
  # Calculate age based on date of birth (dob)
  dob = pd.to_datetime(selected_customer['dob'])
  age = int((pd.Timestamp.now() - dob).days / 365.25)

  # Assign to an age group based on age
  if 18 <= age <= 25:
      age_group = "18-25"
  elif 26 <= age <= 35:
      age_group = "26-35"
  elif 36 <= age <= 45:
      age_group = "36-45"
  elif 46 <= age <= 55:
      age_group = "46-55"
  elif 56 <= age <= 65:
      age_group = "56-65"
  else:
      age_group = "65+"

  # Map age_group to one-hot encoded columns
  age_groups = {
      "18-25": [1, 0, 0, 0, 0, 0],
      "26-35": [0, 1, 0, 0, 0, 0],
      "36-45": [0, 0, 1, 0, 0, 0],
      "46-55": [0, 0, 0, 1, 0, 0],
      "56-65": [0, 0, 0, 0, 1, 0],
      "65+": [0, 0, 0, 0, 0, 1]
  }
  age_group_encoded = age_groups.get(age_group, [0] * 6)

  # Map gender to binary columns
  gender_F = 1 if selected_customer['gender'] == "Female" else 0
  gender_M = 1 if selected_customer['gender'] == "Male" else 0

  # One-hot encode transaction category (example for simplicity)
  categories = [
      'category_food_dining', 'category_gas_transport', 'category_grocery_net',
      'category_grocery_pos', 'category_health_fitness', 'category_home',
      'category_kids_pets', 'category_misc_net', 'category_misc_pos',
      'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
      'category_travel'
  ]
  category = selected_customer['category']
  category_encoded = [1 if category == cat else 0 for cat in categories]
  
  
  # Function to calculate the Haversine distance
  def haversine(lat1, lon1, lat2, lon2):
      # Radius of Earth in miles
      R = 3958.8 
      lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
      dlat = lat2 - lat1
      dlon = lon2 - lon1
      a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
      c = 2 * atan2(sqrt(a), sqrt(1 - a))
      distance = R * c
      return distance
  # Calculate transaction distance
  merch_lat, merch_long = selected_customer['merch_lat'], selected_customer['merch_long']
  lat, long = selected_customer['lat'], selected_customer['long']
  distance = haversine(lat, long, merch_lat, merch_long)

  # Build final input DataFrame for the model
  input_data = {
      'amt': [selected_customer['amt']],
      'zip': [selected_customer['zip']],
      'city_pop': [selected_customer['city_pop']],
      'age': [age],
      'state_code': customer_data['state_code'],
      'hour': [pd.to_datetime(selected_customer['trans_date_trans_time']).hour],
      'day_of_week': [pd.to_datetime(selected_customer['trans_date_trans_time'])
      .weekday() + 1],
      'is_weekend': [1 if pd.to_datetime(selected_customer['trans_date_trans_time'])
      .weekday() >= 5 else 0],
      'is_business_hours': [1 if 9 <= pd.to_datetime(
      selected_customer['trans_date_trans_time']).hour <= 17 else 0],
      **{f'category_{cat}': cat_enc for cat, cat_enc in 
      zip(categories, category_encoded)},
      'gender_F': [gender_F],
      'gender_M': [gender_M],
      **{f'age_group_{age_group_key}': [age_group_val] for age_group_key, age_group_val 
      in zip(age_groups.keys(), age_group_encoded)},
      'distance': [distance]
  }

  return pd.DataFrame(input_data)
# Load the churn data
df = pd.read_csv("churn.csv")
fraud_data = pd.read_csv("fraud_data.csv")

# Set the name of tab
st.set_page_config(page_title="Genos churn")

option = st.sidebar.selectbox(
  "Select a service",
  ("Churn Prediction", "Fraud Detection")
)
if option == "Churn Prediction":

  st.title("Genos Bank customer churn prediction")
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
    if st.button("Predict churn"):
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
elif option == "Fraud Detection":
  st.title("Fraud Detection Analysis")
  # Input fields for transaction information
  customers = [f"{row['cc_num']} - {row['last']}" for _, row in fraud_data.iterrows()]
  selected_customer_option = st.selectbox("Select a transaction", customers)
 
  if selected_customer_option:
      selected_customer_cc_num = int(selected_customer_option.split(" - ")[0])
      # Identify selected transaction row
      selected_customer = fraud_data.loc[fraud_data['cc_num'] == 
      selected_customer_cc_num].iloc[0]
      input_df = prepare_fraud_input(selected_customer)

      # Calculate distance between customer and merchant
      map_data = pd.DataFrame({
        'name': ['Customer', 'Merchant'],
        'lat': [selected_customer['lat'], selected_customer['merch_lat']],
        'lon': [selected_customer['long'], selected_customer['merch_long']]
      })

      # Create the line layer to connect the customer and merchant
      line_data = pd.DataFrame({
        'start_lat': [selected_customer['lat']],
        'start_lon': [selected_customer['long']],
        'end_lat': [selected_customer['merch_lat']],
        'end_lon': [selected_customer['merch_long']]
      })

      # Pydeck Layer setup
      layer_points = pdk.Layer(
        'ScatterplotLayer',
        data=map_data,
        get_position='[lon, lat]',
        get_fill_color='[200, 30, 0, 160]', # Red color
        get_radius=200, # Radius of the points
        pickable=True
      )

      layer_line = pdk.Layer(
        "LineLayer",
        data=line_data,
        get_source_position='[start_lon, start_lat]',
        get_target_position='[end_lon, end_lat]',
        get_color='[0, 0, 255, 160]', # Blue color for the line
        get_width=5
      )

      # Set up the initial view for the map
      view_state = pdk.ViewState(
        latitude=(selected_customer['lat'] + selected_customer['merch_lat']) / 2,
        longitude=(selected_customer['long'] + selected_customer['merch_long']) / 2,
        zoom=10,
        pitch=50
      )

      # Render the map with the points and line layers
      # Render the map with the points and line layers
      tooltip_config = {
          'html': '<b>{name}</b>',
          'style': {
              'color': 'white',
              'backgroundColor': 'red'
          }
      }
      st.pydeck_chart(
          pdk.Deck(
              layers=[layer_points, layer_line],
              initial_view_state=view_state,
              tooltip=tooltip_config,
          )
      )
     
    
      # Prefill transaction fields
      col1, col2 = st.columns(2)
      with col1:
          distance_series = input_df['distance']
          distance_value = int(distance_series.iloc[0])
          distance = st.number_input("Transaction Distance (Miles)",
          value=distance_value)
          amount = st.number_input("Transaction Amount", min_value=0.0,
          value=float(selected_customer['amt']), step=0.01)
          hour = st.number_input("Transaction Hour", min_value=0, max_value=23,
          value=pd.to_datetime(selected_customer['trans_date_trans_time']).hour)
          day_of_week = st.number_input("Day of the Week (1=Monday, 7=Sunday)",
          min_value=1,max_value=7,
          value=pd.to_datetime(selected_customer['trans_date_trans_time']).weekday()
          + 1)
          is_weekend = st.checkbox("Transaction on Weekend",
          value=pd.to_datetime(selected_customer['trans_date_trans_time']).weekday()
          >= 5)
          is_business_hours = st.checkbox("During Business Hours (9 AM - 5 PM)", 
          value=9 <=pd.to_datetime(selected_customer['trans_date_trans_time']).hour
          <= 17)

      with col2:
          state_code = st.text_input("State Code", value=selected_customer['state'])
          job_code = st.text_input("Job Code", value=selected_customer['job'])
          category = st.text_input("Transaction Category",
          value=selected_customer['category'])
          age_value_series = input_df['age']  # This is a Series
          age_value = int(age_value_series.iloc[0])
          gender = st.radio("Gender", ["Male", "Female"],
          index=0 if selected_customer['gender'] == "Male" else 1)
          age = st.number_input("Age", value=age_value)
        

      # Make prediction
      if st.button("Predict Fraud Risk"):
        fraud_probability = fraud_detection_model.predict_proba(input_df)[0][1]
        # Display gauge chart
        fig = create_fraud_gauge_chart(fraud_probability)
        st.plotly_chart(fig, use_container_width=True)