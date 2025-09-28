
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

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
with col2:
    Employee_Status = st.radio("Employment Status", ["Employed", "Unemployed"], horizontal=True)
    Level_of_Education = st.radio("Level of Education", ["Primary", "Secondary", "Tertiary", "Other"], horizontal=True)
    Shopping_frequency = st.radio("Shopping frequency", ["1-2x/week", "2-3x/week", "3-4x/week", "5-6x/week", "6-7x/week"], horizontal=False)
    Regular_Customer = st.radio('Customer Type', ["Regular", "Only when needed"], horizontal=True)
