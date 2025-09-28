# app.py
import sys, platform, pickle, json, os
import numpy as np
import pandas as pd
import streamlit as st

# ------------ Friendly import guard for pgmpy ------------
try:
    from pgmpy.inference import VariableElimination
    from pgmpy.models import BayesianModel
except ModuleNotFoundError:
    st.error(
        "The package **pgmpy** is not installed. Install it and rerun the app."
    )
    st.code(
        "pip install --upgrade pip setuptools wheel\n"
        "pip install pgmpy numpy pandas networkx scipy streamlit",
        language="bash",
    )
    st.stop()

# ------------ App config & styles ------------
PICKLE_PATH = "bn_pgmpy.pkl"
st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛍️", layout="centered")
st.markdown(
    """
    <style>
      .block-container {max-width: 1050px;}
      .metric-card {background: #f8f9fb; padding: 12px 16px; border-radius: 14px; border: 1px solid #eef0f4;}
      .stRadio > label {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🛍️ Purchase Intention (Bayesian Network)")
st.caption(
    f"Python {sys.version.split()[0]} • Platform {platform.platform()}"
)

# ------------ Load bundle ------------
@st.cache_resource(show_spinner=False)
def load_bundle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_bayesian_model(m):
    # Some pickles load a class without .check_model; rebuild as BayesianModel.
    if hasattr(m, "check_model"):
        return m
    try:
        bm = BayesianModel(m.edges())
        bm.add_nodes_from(list(m.nodes()))
        for cpd in m.get_cpds():
            bm.add_cpds(cpd)
        try:
            bm.check_model()
        except Exception as e:
            st.warning(f"Adapted model didn't pass check_model(): {e}")
        return bm
    except Exception as e:
        st.error(f"Could not adapt model to BayesianModel: {e}")
        raise

try:
    bundle = load_bundle(PICKLE_PATH)
except Exception as e:
    st.error(f"❌ Failed to load pickle at '{PICKLE_PATH}': {e}")
    st.stop()

for k in ("model", "target", "classes", "state_names"):
    if k not in bundle:
        st.error(f"❌ Pickle missing required key: {k}")
        st.stop()

bundle["model"] = ensure_bayesian_model(bundle["model"])
model = bundle["model"]
target_node = bundle["target"]
classes = bundle["classes"]
state_names: dict = bundle["state_names"]

# ------------ Helpers ------------
def pick_existing_node(state_names: dict, candidates: list[str]):
    for c in candidates:
        if c in state_names:
            return c
    return None

def clamp_round_1_to_5(x: float) -> int:
    return int(min(5, max(1, round(x))))

def median_from_lookup(medians_dict, demo_name, demo_label_value, fallback=3):
    """
    medians_dict is keyed by friendly labels (e.g., 'Male'), not codes.
    demo_label_value is the friendly label (we convert codes->labels before calling this).
    """
    d = medians_dict.get(demo_name, {})
    # Normalize newline variant for 'Only when needed'
    if demo_label_value == "Only when needed" and "Only when\nneeded" in d:
        demo_label_value = "Only when\nneeded"
    return int(d.get(demo_label_value, fallback))

def averaged_score_for_var(var_medians_dict, demo_answers_labels: dict) -> int:
    """Average across demographics using FRIENDLY LABELS (not codes), then clamp to 1..5."""
    vals = []
    for demo_name, demo_label in demo_answers_labels.items():
        if demo_label is None or demo_label == "":
            continue
        vals.append(median_from_lookup(var_medians_dict, demo_name, demo_label, fallback=3))
    if not vals:
        return 3
    return clamp_round_1_to_5(float(np.mean(vals)))

def predict_purchase(bundle, evidence: dict):
    model = bundle["model"]
    target = bundle["target"]
    classes = bundle["classes"]
    ve = VariableElimination(model)

    q = ve.query([target], evidence=evidence, show_progress=False)
    probs = dict(zip(q.state_names[target], q.values.flatten().astype(float)))
    prob_vec = np.array([probs.get(c, 0.0) for c in classes], dtype=float)
    pred_idx = int(prob_vec.argmax())
    return target, classes[pred_idx], float(prob_vec[pred_idx]), classes, prob_vec

# ------------ Label map (pretty UI ↔ codes the model expects) ------------
LABEL_MAP_PATH = "state_label_map.json"  # optional override next to app.py
DEFAULT_LABELS = {
    # These defaults assume your demographics are encoded as numeric strings "1","2","3", etc.
    # If your pickle stores human words (e.g., "Male"), the labels will fall back to identity.
    "Gender": {"1": "Male", "2": "Female", "3": "Prefer not to say"},
    "Age": {"1": "18–22", "2": "23–28", "3": "29–35", "4": "35–49", "5": "50–65"},
    "Marital_Status": {"1": "Married", "2": "Single", "3": "Prefer not to say"},
    "Employment_Status": {"1": "Employed", "2": "Unemployed"},
    "Level_of_Education": {"1": "Primary", "2": "Secondary", "3": "Tertiary", "4": "Other"},
    "Shopping_frequency": {"1": "1–2x/week", "2": "2–3x/week", "3": "3–4x/week", "4": "5–6x/week", "5": "6–7x/week"},
    "Regular_Customer": {"1": "Regular", "2": "Only when needed"},
    # Likert-like nodes usually already coded "1".."5"—we show richer labels via format_func below.
}

def load_label_map(state_names: dict) -> dict:
    labels = {}
    for node, states in state_names.items():
        base = DEFAULT_LABELS.get(node, {})
        labels[node] = {s: base.get(s, s) for s in states}  # identity fallback
    if os.path.exists(LABEL_MAP_PATH):
        try:
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                override = json.load(f)
            for node, mapping in override.items():
                if node in labels:
                    labels[node].update(mapping)
        except Exception as e:
            st.warning(f"Could not read {LABEL_MAP_PATH}: {e}")
    return labels

LABELS = load_label_map(state_names)

def radio_mapped(title: str, node_key: str, *, horizontal: bool = True):
    """
    Shows pretty labels but returns the BN state CODE (string) for that node.
    If no labels available, shows the raw codes.
    """
    opts = state_names.get(node_key, [])
    if not opts:
        return None
    display = []
    for code in opts:
        lab = LABELS.get(node_key, {}).get(code, code)
        display.append(lab if lab == code else f"{lab} [{code}]")
    idx = st.radio(title, list(range(len(opts))), format_func=lambda i: display[i], horizontal=horizontal)
    return opts[idx]  # <- return the underlying code (string), exactly as the BN expects

def likert_radio(title: str, node_key: str):
    opts = state_names.get(node_key, ["1", "2", "3", "4", "5"])
    labels = {"1": "1 = Strongly disagree", "2": "2 = Disagree", "3": "3 = Indifferent",
              "4": "4 = Agree", "5": "5 = Strongly agree"}
    idx = st.radio(title, list(range(len(opts))), format_func=lambda i: labels.get(opts[i], opts[i]), horizontal=True)
    return opts[idx]

# ------------ Discover node names in the model (underscore/space tolerant) ------------
Gender = pick_existing_node(state_names, ["Gender"])
Age = pick_existing_node(state_names, ["Age"])
Marital_Status = pick_existing_node(state_names, ["Marital_Status", "Marital Status"])
Employment_Status = pick_existing_node(state_names, ["Employment_Status", "Employment Status"])
Level_of_Education = pick_existing_node(state_names, ["Level_of_Education", "Level of Education"])
Regular_Customer = pick_existing_node(state_names, ["Regular_Customer", "Regular Customer", "Customer_Type", "Customer Type"])
Shopping_frequency = pick_existing_node(state_names, ["Shopping_frequency", "Shopping frequency"])

Empathy = pick_existing_node(state_names, ["Empathy"])
Convenience = pick_existing_node(state_names, ["Convenience"])
Customer_Trust = pick_existing_node(state_names, ["Customer_Trust", "Customer Trust"])

Perceived_Value = pick_existing_node(state_names, ["Perceived_Value", "Perceived Value"])
Perceived_Product_Quality = pick_existing_node(state_names, ["Perceived_Product_Quality", "Perceived Product Quality"])
Physical_Environment = pick_existing_node(state_names, ["Physical_Environment", "Physical Environment"])
Price_Sensitivity = pick_existing_node(state_names, ["Price_Sensitivity", "Price Sensitivity"])

# ------------ Medians (by FRIENDLY LABELS) for auto-computed variables ------------
empathy_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18–22": 3, "23–28": 4, "29–35": 4, "35–49": 3, "50–65": 4},
    "Marital_Status": {"Married": 4, "Single": 4, "Prefer not to say": 3},
    "Shopping_frequency": {"1–2x/week": 3, "2–3x/week": 4, "3–4x/week": 4, "5–6x/week": 4, "6–7x/week": 4},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 4, "Unemployed": 4},
}
convenience_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18–22": 3, "23–28": 4, "29–35": 4, "35–49": 4, "50–65": 4},
    "Marital_Status": {"Married": 4, "Single": 4, "Prefer not to say": 5},
    "Shopping_frequency": {"1–2x/week": 4, "2–3x/week": 4, "3–4x/week": 5, "5–6x/week": 5, "6–7x/week": 2},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 4, "Unemployed": 4},
}
customer_trust_medians = {
    "Gender": {"Male": 4, "Female": 3, "Prefer not to say": 3},
    "Age": {"18–22": 4, "23–28": 4, "29–35": 3, "35–49": 4, "50–65": 3},
    "Marital_Status": {"Married": 4, "Single": 3, "Prefer not to say": 3},
    "Shopping_frequency": {"1–2x/week": 3, "2–3x/week": 4, "3–4x/week": 3, "5–6x/week": 5, "6–7x/week": 3},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 3, "Unemployed": 4},
}

# ------------ UI: Demographics ------------
st.header("1) Demographics")

col1, col2 = st.columns(2, gap="large")
with col1:
    ui_gender_code = radio_mapped("Gender", Gender, horizontal=True)
    ui_age_code = radio_mapped("Age", Age, horizontal=True)
    ui_marital_code = radio_mapped("Marital Status", Marital_Status, horizontal=True)
with col2:
    ui_empstat_code = radio_mapped("Employment Status", Employment_Status, horizontal=True)
    ui_edu_code = radio_mapped("Level of Education", Level_of_Education, horizontal=True)
    ui_shopfreq_code = radio_mapped("Shopping frequency", Shopping_frequency, horizontal=False)
    ui_regular_code = radio_mapped("Customer Type", Regular_Customer, horizontal=True)

# Convert codes -> friendly labels for the medians computation
def label_of(node_key, code):
    return LABELS.get(node_key, {}).get(code, code)

demo_answers_labels = {
    "Gender": label_of(Gender, ui_gender_code) if ui_gender_code else None,
    "Age": label_of(Age, ui_age_code) if ui_age_code else None,
    "Marital_Status": label_of(Marital_Status, ui_marital_code) if ui_marital_code else None,
    "Employment_Status": label_of(Employment_Status, ui_empstat_code) if ui_empstat_code else None,
    "Level_of_Education": label_of(Level_of_Education, ui_edu_code) if ui_edu_code else None,
    "Shopping_frequency": label_of(Shopping_frequency, ui_shopfreq_code) if ui_shopfreq_code else None,
    "Regular_Customer": label_of(Regular_Customer, ui_regular_code) if ui_regular_code else None,
}

# ------------ Auto-computed latent variables ------------
st.header("2) Auto-computed (from demographics)")
emp_score = averaged_score_for_var(empathy_medians, demo_answers_labels)
conv_score = averaged_score_for_var(convenience_medians, demo_answers_labels)
trust_score = averaged_score_for_var(customer_trust_medians, demo_answers_labels)

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

# ------------ Likert questions (return codes as strings) ------------
st.header("3) Customer answers (1–5)")
q_val_code = likert_radio("Perceived Value", Perceived_Value)
q_qual_code = likert_radio("Perceived Product Quality", Perceived_Product_Quality)
q_env_code = likert_radio("Physical Environment", Physical_Environment)
q_price_code = likert_radio("Price Sensitivity", Price_Sensitivity)

# ------------ Build evidence (send CODES to BN, strings only) ------------
evidence = {}
def put(node_key, code_str):
    if node_key and code_str is not None:
        evidence[node_key] = str(code_str)

put(Gender, ui_gender_code)
put(Age, ui_age_code)
put(Marital_Status, ui_marital_code)
put(Employment_Status, ui_empstat_code)
put(Level_of_Education, ui_edu_code)
put(Shopping_frequency, ui_shopfreq_code)
put(Regular_Customer, ui_regular_code)

put(Empathy, str(emp_score))
put(Convenience, str(conv_score))
put(Customer_Trust, str(trust_score))

put(Perceived_Value, q_val_code)
put(Perceived_Product_Quality, q_qual_code)
put(Physical_Environment, q_env_code)
put(Price_Sensitivity, q_price_code)

# ------------ Predict ------------
st.header("4) Prediction")
if st.button("Predict Purchase Intention"):
    try:
        tgt, pred_class, conf, cls, prob_vec = predict_purchase(bundle, evidence)
        st.success(f"Predicted **{tgt}**: **{pred_class}**  |  Confidence: **{conf*100:.1f}%**")

        prob_df = pd.DataFrame({"Class": cls, "Probability": prob_vec})
        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("Show evidence used"):
            st.json(evidence)

        with st.expander("Debug: label map for demographics"):
            for node in [Gender, Age, Marital_Status, Employment_Status, Level_of_Education, Shopping_frequency, Regular_Customer]:
                if node:
                    st.write(f"**{node}**")
                    st.table(pd.DataFrame({"code": state_names[node], 
                                           "label": [LABELS.get(node, {}).get(s, s) for s in state_names[node]]}))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Evidence (debug)"):
            st.json(evidence)
