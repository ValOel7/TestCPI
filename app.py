
import streamlit as st
import pickle
import numpy as np
import pandas as pd

PICKLE_PATH = "bn_pgmpy.pkl"   


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛍️", layout="centered")
st.title("Purchase Intention (Bayesian Network)")
st.caption("Enter demographics and the answers to the four underlying questions based on a scale of 1–5. The app computes Empathy, Convenience, and Customer Trust from demographics, then predicts Purchase Intention.")

# Demographic inputs
st.header("1) Demographics")
col1, col2 = st.columns(2)
with col1:
    Gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"], horizontal=True)
    Age = st.radio("Age", ["18-22", "23-28", "29-35", "35-49", "50-65"], horizontal=True)
    Marital_Status = st.radio("Marital Status", ["Married", "Single", "Prefer not to say"], horizontal=True)
    Regular_Customer = st.radio('Customer Type', ["Regular", "Only when needed"], horizontal=True)
with col2:
    Employee_Status = st.radio("Employment Status", ["Employed", "Unemployed"], horizontal=True)
    Level_of_Education = st.radio("Level of Education", ["Primary", "Secondary", "Tertiary", "Other"], horizontal=True)
    Shopping_frequency = st.radio("Shopping frequency", ["1-2x/week", "2-3x/week", "3-4x/week", "5-6x/week", "6-7x/week"], horizontal=False)

# Load the trained model
with open('bn_pgmpy.pkl', 'rb') as file:
    model = pickle.load(file)
    
TARGET = bundle["target"]

demo_answers = {
    "Gender": Gender,
    "Age": Age,
    "Marital Status": Marital_Status,
    "Employment Status": Employee_Status,
    "Level of Education": Level_of_Education,
    "Shopping frequency": Shopping_frequency,
    "Regular Customer": Regular_Customer,
}

# Compute the 3 biased variables from demographics
st.header("2) Auto-computed (from demographics)")
# Empathy: use provided empathy_medians if available; otherwise neutral 3s
emp_score = averaged_score_for_var(empathy_medians, demo_answers)
conv_score = averaged_score_for_var(convenience_medians, demo_answers)
trust_score = averaged_score_for_var(customer_trust_medians, demo_answers)

colA, colB, colC = st.columns(3)
colA.metric("Empathy (1–5)", emp_score)
colB.metric("Convenience (1–5)", conv_score)
colC.metric("Customer Trust (1–5)", trust_score)

# Likert labels
LIKERT_LABELS = {
    "1": "1 = Strongly disagree",
    "2": "2 = Disagree",
    "3": "3 = Indifferent",
    "4": "4 = Agree",
    "5": "5 = Strongly agree",
}
LIKERT_OPTS = ["1", "2", "3", "4", "5"]

st.header("3) Ask 4 questions (customer answers)")
q1 = st.radio("Perceived Value", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)
q2 = st.radio("Perceived Product Quality", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)
q3 = st.radio("Physical Environment", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)
q4 = st.radio("Price Sensitivity", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)

# Build evidence dict (convert everything to strings as BN was trained on strings)
# IMPORTANT: variable names must match the BN node names in your training data.
evidence = {
    # demographics
    "Gender": Gender,
    "Age": Age,
    "Marital Status": Marital_Status,
    "Employment Status": Employee_Status,
    "Level of Education": Level_of_Education,
    "Shopping frequency": Shopping_frequency,
    "Regular Customer": "Only when\nneeded" if Regular_Customer == "Only when needed" else Regular_Customer,

    # three biased vars (as strings)
    "Empathy": str(emp_score),
    "Convenience": str(conv_score),
    "Customer Trust": str(trust_score),

    # four customer-answered vars (already strings 1..5)
    "Perceived Value": q1,
    "Perceived Product Quality": q2,
    "Physical Environment": q3,
    "Price Sensitivity": q4,
}

st.header("4) Prediction")
if st.button("Predict Purchase Intention"):
    try:
        pred_class, conf, classes, prob_vec = predict_purchase(bundle, evidence)
        st.success(f"Predicted **{TARGET}**: **{pred_class}**  |  Confidence: **{conf*100:.1f}%**")

        # Show probability distribution
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": prob_vec
        })
        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("Show evidence used"):
            st.json(evidence)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Evidence (debug)"):
            st.json(evidence)
