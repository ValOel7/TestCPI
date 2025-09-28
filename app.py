# app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pgmpy.inference import VariableElimination

# ---------- SETTINGS ----------
PICKLE_PATH = "bn_pgmpy.pkl"  

# Lookup medians (used to auto-compute 3 latent vars)
empathy_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 3, "50-65": 4},
    "Marital_Status": {"Married": 4, "Single": 4, "Prefer not to say": 3},
    "Shopping_frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 4, "5-6x/week": 4, "6-7x/week": 4},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 4, "Unemployed": 4}
}
convenience_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 4, "50-65": 4},
    "Marital_Status": {"Married": 4, "Single": 4, "Prefer not to say": 5},
    "Shopping_frequency": {"1-2x/week": 4, "2-3x/week": 4, "3-4x/week": 5, "5-6x/week": 5, "6-7x/week": 2},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 4, "Unemployed": 4},
}
customer_trust_medians = {
    "Gender": {"Male": 4, "Female": 3, "Prefer not to say": 3},
    "Age": {"18-22": 4, "23-28": 4, "29-35": 3, "35-49": 4, "50-65": 3},
    "Marital_Status": {"Married": 4, "Single": 3, "Prefer not to say": 3},
    "Shopping_frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 3, "5-6x/week": 5, "6-7x/week": 3},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 3, "Unemployed": 4},
}

LIKERT_OPTS = ["1", "2", "3", "4", "5"]
LIKERT_LABELS = {
    "1": "1 = Strongly disagree",
    "2": "2 = Disagree",
    "3": "3 = Indifferent",
    "4": "4 = Agree",
    "5": "5 = Strongly agree",
}

st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛍️", layout="centered")
st.markdown(
    """
    <style>
    .block-container {max-width: 1050px;}
    .stRadio > label {font-weight: 600;}
    .metric-card {background: #f8f9fb; padding: 12px 16px; border-radius: 14px; border: 1px solid #eef0f4;}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🛍️ Purchase Intention (Bayesian Network)")
st.caption("Provide demographics and 4 answers (1–5). The app auto-computes Empathy, Convenience, Customer Trust from demographics, then predicts Purchase Intention.")

# --------- Utils ----------
@st.cache_resource(show_spinner=False)
def load_bundle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def clamp_round_1_to_5(x: float) -> int:
    return int(min(5, max(1, round(x))))

def median_from_lookup(medians_dict, demo_name, demo_value, fallback=3):
    if medians_dict is None:
        return int(fallback)
    d = medians_dict.get(demo_name, {})
    # normalize "Only when needed" variants
    if demo_value == "Only when needed":
        demo_value = "Only when\nneeded" if "Only when\nneeded" in d else "Only when needed"
    return int(d.get(demo_value, fallback))

def averaged_score_for_var(var_medians_dict, demo_answers: dict) -> int:
    vals = []
    for demo_name, demo_value in demo_answers.items():
        if demo_value is None or demo_value == "":
            continue
        vals.append(median_from_lookup(var_medians_dict, demo_name, demo_value, fallback=3))
    if not vals:
        return 3
    return clamp_round_1_to_5(float(np.mean(vals)))

def pick_existing_node(state_names: dict, candidates: list[str]) -> str | None:
    """Return the first candidate that exists in state_names (handles underscore vs space names)."""
    for c in candidates:
        if c in state_names:
            return c
    return None

def predict_purchase(bundle, evidence: dict):
    model = bundle["model"]
    target = bundle["target"]
    classes = bundle["classes"]
    state_names = bundle["state_names"]

    ve = VariableElimination(model)
    nodes = set(model.nodes())
    valid = {v: set(s) for v, s in state_names.items()}

    # keep only valid evidence and states, coerce to str
    ev = {}
    for k, v in evidence.items():
        if k in nodes:
            v = str(v)
            # allow unseen but don't crash: if model has valid states, keep only valid ones
            if k in valid and v not in valid[k]:
                # try newline normalization for "Only when needed"
                if "Only when\nneeded" in valid[k] and v == "Only when needed":
                    v = "Only when\nneeded"
                elif "Only when needed" in valid[k] and v == "Only when\nneeded":
                    v = "Only when needed"
            if (k not in valid) or (v in valid[k]):
                ev[k] = v

    q = ve.query([target], evidence=ev, show_progress=False)
    probs = dict(zip(q.state_names[target], q.values.flatten().astype(float)))

    prob_vec = np.array([probs.get(c, 0.0) for c in classes], dtype=float)
    pred_idx = int(prob_vec.argmax())
    pred_class = classes[pred_idx]
    confidence = float(prob_vec[pred_idx])
    return target, pred_class, confidence, classes, prob_vec, ev

# --------- Load model & discover schema ----------
try:
    bundle = load_bundle(PICKLE_PATH)
except Exception as e:
    st.error(f"❌ Failed to load pickle at '{PICKLE_PATH}': {e}")
    st.stop()

required_keys = ["model", "target", "classes", "state_names"]
missing = [k for k in required_keys if k not in bundle]
if missing:
    st.error(f"❌ Pickle is missing keys: {missing}")
    st.stop()

state_names: dict = bundle["state_names"]
target_node = bundle["target"]

# Provide resilient names: try underscore names first (your new columns), fall back to space names
Gender = pick_existing_node(state_names, ["Gender"])
Age = pick_existing_node(state_names, ["Age"])
Marital_Status = pick_existing_node(state_names, ["Marital_Status", "Marital Status"])
Employment_Status = pick_existing_node(state_names, ["Employment_Status", "Employment Status"])
Level_of_Education = pick_existing_node(state_names, ["Level_of_Education", "Level of Education"])
Regular_Customer = pick_existing_node(state_names, ["Regular_Customer", "Customer Type", "Customer_Type", "Regular Customer"])
Shopping_frequency = pick_existing_node(state_names, ["Shopping_frequency", "Shopping frequency"])

Empathy = pick_existing_node(state_names, ["Empathy"])
Convenience = pick_existing_node(state_names, ["Convenience"])
Customer_Trust = pick_existing_node(state_names, ["Customer_Trust", "Customer Trust"])

Perceived_Value = pick_existing_node(state_names, ["Perceived_Value", "Perceived Value"])
Perceived_Product_Quality = pick_existing_node(state_names, ["Perceived_Product_Quality", "Perceived Product Quality"])
Physical_Environment = pick_existing_node(state_names, ["Physical_Environment", "Physical Environment"])
Price_Sensitivity = pick_existing_node(state_names, ["Price_Sensitivity", "Price Sensitivity"])

# ---------- UI: Demographics ----------
st.header("1) Demographics")

col1, col2 = st.columns(2, gap="large")

with col1:
    gender_opt = state_names.get(Gender, ["Male", "Female", "Prefer not to say"])
    age_opt = state_names.get(Age, ["18-22", "23-28", "29-35", "35-49", "50-65"])
    marital_opt = state_names.get(Marital_Status, ["Married", "Single", "Prefer not to say"])

    ui_gender = st.radio("Gender", gender_opt, horizontal=True)
    ui_age = st.radio("Age", age_opt, horizontal=True)
    ui_marital = st.radio("Marital Status", marital_opt, horizontal=True)

with col2:
    empstat_opt = state_names.get(Employment_Status, ["Employed", "Unemployed"])
    edu_opt = state_names.get(Level_of_Education, ["Primary", "Secondary", "Tertiary", "Other"])
    shopfreq_opt = state_names.get(Shopping_frequency, ["1-2x/week", "2-3x/week", "3-4x/week", "5-6x/week", "6-7x/week"])
    regular_opt = state_names.get(Regular_Customer, ["Regular", "Only when\nneeded"])

    ui_empstat = st.radio("Employment Status", empstat_opt, horizontal=True)
    ui_edu = st.radio("Level of Education", edu_opt, horizontal=True)
    ui_shopfreq = st.radio("Shopping frequency", shopfreq_opt, horizontal=False)
    ui_regular = st.radio("Customer Type", regular_opt, horizontal=True)

demo_answers = {
    "Gender": ui_gender,
    "Age": ui_age,
    "Marital_Status": ui_marital,
    "Employment_Status": ui_empstat,
    "Level_of_Education": ui_edu,
    "Shopping_frequency": ui_shopfreq,
    "Regular_Customer": ui_regular,
}

# ---------- Auto-computed latent scores ----------
st.header("2) Auto-computed (from demographics)")

emp_score = averaged_score_for_var(empathy_medians, demo_answers)
conv_score = averaged_score_for_var(convenience_medians, demo_answers)
trust_score = averaged_score_for_var(customer_trust_medians, demo_answers)

colA, colB, colC = st.columns(3)
with colA:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Empathy (1–5)", emp_score)
    st.markdown("</div>", unsafe_allow_html=True)
with colB:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Convenience (1–5)", conv_score)
    st.markdown("</div>", unsafe_allow_html=True)
with colC:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Customer Trust (1–5)", trust_score)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Likert questions ----------
st.header("3) Customer answers (1–5)")
q1 = st.radio("Perceived Value", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)
q2 = st.radio("Perceived Product Quality", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)
q3 = st.radio("Physical Environment", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)
q4 = st.radio("Price Sensitivity", LIKERT_OPTS, format_func=lambda x: LIKERT_LABELS[x], horizontal=True)

# ---------- Evidence builder ----------
# Build evidence using the discovered node names. Only include keys that exist in the model.
evidence = {}
def put_if_present(node_key, value):
    if node_key:  # only if node exists in BN
        evidence[node_key] = str(value)

put_if_present(Gender, ui_gender)
put_if_present(Age, ui_age)
put_if_present(Marital_Status, ui_marital)
put_if_present(Employment_Status, ui_empstat)
put_if_present(Level_of_Education, ui_edu)
put_if_present(Shopping_frequency, ui_shopfreq)
# normalize "Only when needed" if model uses newline
if Regular_Customer:
    rc_val = ui_regular
    valid_states = set(state_names.get(Regular_Customer, []))
    if "Only when\nneeded" in valid_states and rc_val == "Only when needed":
        rc_val = "Only when\nneeded"
    if "Only when needed" in valid_states and rc_val == "Only when\nneeded":
        rc_val = "Only when needed"
    put_if_present(Regular_Customer, rc_val)

# Auto-computed latent variables
put_if_present(Empathy, str(emp_score))
put_if_present(Convenience, str(conv_score))
put_if_present(Customer_Trust, str(trust_score))

# Likert variables
put_if_present(Perceived_Value, q1)
put_if_present(Perceived_Product_Quality, q2)
put_if_present(Physical_Environment, q3)
put_if_present(Price_Sensitivity, q4)

# ---------- Predict ----------
st.header("4) Prediction")
if st.button("Predict Purchase Intention"):
    try:
        tgt, pred_class, conf, classes, prob_vec, ev_used = predict_purchase(bundle, evidence)
        st.success(f"Predicted **{tgt}**: **{pred_class}**  |  Confidence: **{conf*100:.1f}%**")

        prob_df = pd.DataFrame({"Class": classes, "Probability": prob_vec})
        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("Show evidence used"):
            st.json(ev_used)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Evidence (debug)"):
            st.json(evidence)
