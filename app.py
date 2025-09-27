import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pgmpy.inference import VariableElimination

# -----------------------------
# CONFIG: file path to your pickle bundle (model + metadata)
# -----------------------------
PICKLE_PATH = "bn_pgmpy.pkl"   # <-- change if needed

# -----------------------------
# YOUR LOOKUP TABLES (medians)
# NOTE: These are used ONLY to compute the biased defaults for:
#   - Empathy
#   - Convenience
#   - Customer Trust
# by averaging all 7 demographics (if present) and rounding to 1..5.
# If a demographic dimension is missing in a dict, it’s just skipped.
# -----------------------------

# If you have an Empathy-by-demographics table from your violin analysis,
# paste it here. Otherwise we'll fall back to a neutral "3" per demographic.
empathy_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 3, "50-65": 4},
    "Marital Status": {"Married": 4, "Single": 4, "Prefer not to say": 3},
    "Shopping frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 4, "5-6x/week": 4, "6-7x/week": 4},
    "Regular Customer": {"Regular": 4, "Only when\nneeded": 3},
    "Employment Status": {"Employed": 4, "Unemployed": 4}
}
# Fallback: if empathy_medians is empty or missing a demographic/category,
# we'll substitute "3" for that item.

convenience_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 4, "50-65": 4},
    "Marital Status": {"Married": 4, "Single": 4, "Prefer not to say": 5},
    "Shopping frequency": {"1-2x/week": 4, "2-3x/week": 4, "3-4x/week": 5, "5-6x/week": 5, "6-7x/week": 2},
    "Regular Customer": {"Regular": 4, "Only when\nneeded": 3},
    "Employment Status": {"Employed": 4, "Unemployed": 4},
    # "Level of Education": {...}  # add if available
}

customer_trust_medians = {
    "Gender": {"Male": 4, "Female": 3, "Prefer not to say": 3},
    "Age": {"18-22": 4, "23-28": 4, "29-35": 3, "35-49": 4, "50-65": 3},
    "Marital Status": {"Married": 4, "Single": 3, "Prefer not to say": 3},
    "Shopping frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 3, "5-6x/week": 5, "6-7x/week": 3},
    "Regular Customer": {"Regular": 4, "Only when\nneeded": 3},
    "Employment Status": {"Employed": 3, "Unemployed": 4},
    # "Level of Education": {...}
}

# Optional: if you ever want to default the 4 asked variables by demographics,
# you already have these (but in this app we take the user's explicit answers):
perceived_value_medians = {
    "Gender": {"Male": 3, "Female": 3, "Prefer not to say": 3},
    "Age": {"18-22": 2, "23-28": 3, "29-35": 3, "35-49": 4, "50-65": 4},
    "Marital Status": {"Married": 3, "Single": 3, "Prefer not to say": 2},
    "Shopping frequency": {"1-2x/week": 3, "2-3x/week": 3, "3-4x/week": 4, "5-6x/week": 4, "6-7x/week": 2},
    "Regular Customer": {"Regular": 4, "Only when\nneeded": 3},
    "Employment Status": {"Employed": 3, "Unemployed": 3},
}
price_sensitivity_medians = {
    "Gender": {"Male": 3, "Female": 3, "Prefer not to say": 3},
    "Age": {"18-22": 2, "23-28": 3, "29-35": 3, "35-49": 4, "50-65": 4},
    "Marital Status": {"Married": 3, "Single": 3, "Prefer not to say": 3},
    "Shopping frequency": {"1-2x/week": 3, "2-3x/week": 3, "3-4x/week": 4, "5-6x/week": 4, "6-7x/week": 2},
    "Regular Customer": {"Regular": 3, "Only when\nneeded": 3},
    "Employment Status": {"Employed": 3, "Unemployed": 3},
}
perceived_product_quality_medians = {
    "Gender": {"Male": 3, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 4, "50-65": 4},
    "Marital Status": {"Married": 4, "Single": 4, "Prefer not to say": 3},
    "Shopping frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 4, "5-6x/week": 4, "6-7x/week": 3},
    "Regular Customer": {"Regular": 4, "Only when\nneeded": 3},
    "Employment Status": {"Employed": 4, "Unemployed": 3},
}

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_resource
def load_bundle(path: str):
    with open(path, "rb") as f:
        b = pickle.load(f)
    return b

def clamp_round_1_to_5(x: float) -> int:
    return int(min(5, max(1, round(x))))

def median_from_lookup(medians_dict, demo_name, demo_value, fallback=3):
    """Return integer median for (demo_name, demo_value) if present; else fallback."""
    if medians_dict is None:
        return int(fallback)
    d = medians_dict.get(demo_name, {})
    # handle newline variant
    if demo_value not in d and demo_value == "Only when needed":
        demo_value = "Only when\nneeded"
    return int(d.get(demo_value, fallback))

def averaged_score_for_var(var_medians_dict, demo_answers: dict) -> int:
    """Average across up to 7 demographics, round to 1..5. Missing dims are skipped."""
    vals = []
    for demo_name, demo_value in demo_answers.items():
        if demo_value is None or demo_value == "":
            continue
        vals.append(median_from_lookup(var_medians_dict, demo_name, demo_value, fallback=3))
    if not vals:
        return 3
    return clamp_round_1_to_5(np.mean(vals))

def predict_purchase(bundle, evidence: dict):
    """Run pgmpy VE query with evidence filtered to valid nodes/states."""
    model = bundle["model"]
    target = bundle["target"]
    classes = bundle["classes"]
    state_names = bundle["state_names"]

    ve = VariableElimination(model)
    nodes = set(model.nodes())
    valid = {v: set(s) for v, s in state_names.items()}

    # ensure strings
    ev = {k: str(v) for k, v in evidence.items() if k in nodes and str(v) in valid.get(k, {str(v)})}
    q = ve.query([target], evidence=ev, show_progress=False)
    probs = dict(zip(q.state_names[target], q.values.flatten().astype(float)))

    # align probs to class order
    prob_vec = np.array([probs.get(c, 0.0) for c in classes], dtype=float)
    pred_idx = int(prob_vec.argmax())
    pred_class = classes[pred_idx]
    confidence = float(prob_vec[pred_idx])  # 0..1
    return pred_class, confidence, classes, prob_vec

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🧠", layout="centered")
st.title("Purchase Intention (Bayesian Network)")
st.caption("Enter demographics and four answers (1–5). The app computes Empathy, Convenience, and Customer Trust from demographics, then predicts Purchase Intention.")

# Load model
bundle = load_bundle(PICKLE_PATH)
TARGET = bundle["target"]

# Demographic inputs
st.header("1) Demographics")
col1, col2 = st.columns(2)
with col1:
    gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"], horizontal=True)
    age = st.radio("Age", ["18-22", "23-28", "29-35", "35-49", "50-65"], horizontal=True)
    marital = st.radio("Marital Status", ["Married", "Single", "Prefer not to say"], horizontal=True)
with col2:
    emp_status = st.radio("Employment Status", ["Employed", "Unemployed"], horizontal=True)
    edu = st.radio("Level of Education", ["Primary", "Secondary", "Tertiary", "Other"], horizontal=True)
    shop_freq = st.radio("Shopping frequency", ["1-2x/week", "2-3x/week", "3-4x/week", "5-6x/week", "6-7x/week"], horizontal=False)
reg_cust = st.radio('Customer Type', ["Regular", "Only when needed"], horizontal=True)

demo_answers = {
    "Gender": gender,
    "Age": age,
    "Marital Status": marital,
    "Employment Status": emp_status,
    "Level of Education": edu,
    "Shopping frequency": shop_freq,
    "Regular Customer": reg_cust,
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
    "Gender": gender,
    "Age": age,
    "Marital Status": marital,
    "Employment Status": emp_status,
    "Level of Education": edu,
    "Shopping frequency": shop_freq,
    "Regular Customer": "Only when\nneeded" if reg_cust == "Only when needed" else reg_cust,

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
