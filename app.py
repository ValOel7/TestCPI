
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛍️", layout="centered")
st.title("Purchase Intention (Bayesian Network)")
st.caption("Enter demographics and the answers to the four underlying questions based on a scale of 1–5. The app computes Empathy, Convenience, and Customer Trust from demographics, then predicts Purchase Intention.")

