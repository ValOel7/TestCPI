#Import the necessary files
import sys, platform, pickle, json, os
import numpy as np
import pandas as pd
import streamlit as st

#If the purchase intention is 60 and above then the customer will be asked which product category they are interested in and then a specialist staff member will assist the customer
#If purchase intention is less, then a staff member will not be sent out the customer and rather wait for next highly probable purchasing customer
#staff member is sent to assiss the customer in terms of advising on best breads or best diary products as per customer query 
# ===== Assisted-selling config (does NOT affect the BN) =====
ASSIST_THRESHOLD = 0.60  # 60%

PRODUCT_CATEGORIES = [
    "Dairy", "Meat", "Fruit & Veg", "Frozen Foods", "Bakery",
    "Beverages", "Household", "Personal Care", "Snacks & Confectionery", "Pantry / Dry Goods"
]

# Map each category to a small roster; feel free to change names
STAFF_ROSTER = {
    "Dairy": ["Nomsa", "Lerato", "Johan"],
    "Meat": ["Sipho", "Thandi", "Pieter"],
    "Fruit & Veg": ["Ayesha", "Mpho", "Chris"],
    "Frozen Foods": ["Zanele", "Andile", "Nadia"],
    "Bakery": ["Carla", "Neo", "Hendrik"],
    "Beverages": ["Kayla", "Sizwe", "Ruan"],
    "Household": ["Retha", "Kabelo", "Yusuf"],
    "Personal Care": ["Naledi", "Marius", "Anita"],
    "Snacks & Confectionery": ["Bianca", "Xolani", "Gugu"],
    "Pantry / Dry Goods": ["Leah", "Dumisani", "Ben"]
}

def suggest_staff(category: str) -> str:
    """Pick a staff member deterministically based on category, so it doesn't flicker between reruns."""
    roster = STAFF_ROSTER.get(category, [])
    if not roster:
        return "Any available associate"
    idx = (hash(category) % len(roster))
    return roster[idx]

#Import VE as only importing the pickle, not rebuilding the model.
try:
    from pgmpy.inference import VariableElimination
except ModuleNotFoundError:
    st.error("pgmpy is not installed.")
    st.code(
        "pip install --upgrade pip setuptools wheel\n"
        "pip install pgmpy numpy pandas networkx scipy streamlit",
        language="bash",
    )
    st.stop()
#I added the pickle file in the github, so in here it will just be uploaded
PICKLE_PATH = "bn_pgmpy.pkl"

# ---------- Page + styles ----------
st.set_page_config(page_title="Purchase Intention Real-Time Assistant", page_icon="🛍️", layout="centered")
st.markdown(
    """
    <style>
      :root { --accent:#2563eb; }
      .block-container {max-width: 980px;}
      h1, .stMarkdown h1 { letter-spacing: -0.3px; }
      h2, .stMarkdown h2 { margin-top: 0.25rem; }
      .section { background:#fbfcfe; border:1px solid #eef0f4; border-radius:14px; padding:18px 18px 12px; margin: 14px 0 10px; }
      .section h2 { margin: 0 0 14px 0; }
      .metric-card {background:#f8f9fb; padding:12px 16px; border-radius:14px; border:1px solid #eef0f4;}
      .stRadio > label {font-weight:600;}
      .st-emotion-cache-16idsys p { margin-bottom: 0.35rem; } /* tighten labels */
      input[type="radio"]:checked { accent-color: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛍️ Purchase Intention Real-Time Assistant")

#streamlit resource cache to only read the pickle once per session not every rerun
# ---------- load the bundle ----------
@st.cache_resource(show_spinner=False)
def load_bundle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle(PICKLE_PATH)
except Exception as e:
    st.error(f"❌ Failed to load pickle at '{PICKLE_PATH}': {e}")
    st.stop()

# basic key checks
for k in ("model", "target", "classes", "state_names"):
    if k not in bundle:
        st.error(f"❌ Pickle missing required key: {k}")
        st.stop()

model = bundle["model"]
target_node = bundle["target"]
classes = bundle["classes"]
state_names = bundle["state_names"]  # typically: dict node -> list of states

#if the versions of pgmpy change this would be able to still make it work =
# 1) Some VE versions call model.check_model()
if not hasattr(model, "check_model"):
    def _noop_check_model(*args, **kwargs):
        return True
    try:
        model.check_model = _noop_check_model
    except Exception:
        pass

# 2) Some VE versions expect model.factors; derive from CPDs if missing
if not hasattr(model, "factors"):
    try:
        cpds = model.get_cpds()
    except Exception:
        cpds = []
    factors = []
    for cpd in cpds:
        tf = getattr(cpd, "to_factor", None)
        if callable(tf):
            try:
                factors.append(tf()); continue
            except Exception:
                pass
        factors.append(cpd)
    try:
        model.factors = factors
    except Exception:
        pass

# ---------- helpers ----------
def pick_existing_node(state_names_dict, candidates: list[str]):
    if isinstance(state_names_dict, dict):
        for c in candidates:
            if c in state_names_dict:
                return c
    return None

def clamp_round_1_to_5(x: float) -> int:
    return int(min(5, max(1, round(x))))

def median_from_lookup(medians_dict, demo_name, demo_label_value, fallback=3):
    d = medians_dict.get(demo_name, {})
    if demo_label_value == "Only when needed" and "Only when\nneeded" in d:
        demo_label_value = "Only when\nneeded"
    return int(d.get(demo_label_value, fallback))

def averaged_score_for_var(var_medians_dict, demo_answers_labels: dict) -> int:
    vals = []
    for demo_name, demo_label in demo_answers_labels.items():
        if demo_label:
            vals.append(median_from_lookup(var_medians_dict, demo_name, demo_label, fallback=3))
    if not vals:
        return 3
    return clamp_round_1_to_5(float(np.mean(vals)))

def predict_purchase(bundle_obj, evidence: dict):
    ve = VariableElimination(bundle_obj["model"])
    q = ve.query([bundle_obj["target"]], evidence=evidence, show_progress=False)
    probs = dict(zip(q.state_names[bundle_obj["target"]], q.values.flatten().astype(float)))
    prob_vec = np.array([probs.get(c, 0.0) for c in bundle_obj["classes"]], dtype=float)
    idx = int(prob_vec.argmax())
    return bundle_obj["target"], bundle_obj["classes"][idx], float(prob_vec[idx]), bundle_obj["classes"], prob_vec

# ---------- label mapping (pretty UI ↔ string codes the model expects) ----------
LABEL_MAP_PATH = "state_label_map.json"  # optional override
DEFAULT_LABELS = {
    "Gender": {"1": "Male", "2": "Female", "3": "Prefer not to say"},
    "Age": {"1": "18-22", "2": "23-28", "3": "29-35", "4": "35-49", "5": "50-65"},
    "Marital_Status": {"1": "Married", "2": "Single", "3": "Prefer not to say"},
    "Employment_Status": {"1": "Employed", "2": "Unemployed", 1: "Employed", 2: "Unemployed"},
    # Updated per your training mapping (include string and int keys, just in case)
    "Level_of_Education": {
        "1": "No formal", "2": "Basic", "3": "Diploma", "4": "Degree", "5": "Postgrad",
        1: "No formal", 2: "Basic", 3: "Diploma", 4: "Degree", 5: "Postgrad",
    },
    "Shopping_frequency": {"1": "1-2x/week", "2": "2-3x/week", "3": "3-4x/week", "4": "5-6x/week", "5": "6-7x/week"},
    "Regular_Customer": {"1": "Regular", "2": "Only when needed"},
    # Likert nodes already "1".."5"; formatted separately.
}

def load_label_map(state_names_dict) -> dict:
    labels = {}
    if isinstance(state_names_dict, dict):
        for node, states in state_names_dict.items():
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
    """Show pretty labels only; return the underlying BN state code (string)."""
    opts = state_names.get(node_key, []) if isinstance(state_names, dict) else []
    if not opts:
        return None
    pretty = [LABELS.get(node_key, {}).get(code, code) for code in opts]
    idx = st.radio(title, list(range(len(opts))), format_func=lambda i: pretty[i], horizontal=horizontal)
    return opts[idx]  # underlying string code

def likert_radio(title: str, node_key: str):
    opts = state_names.get(node_key, ["1", "2", "3", "4", "5"]) if isinstance(state_names, dict) else ["1","2","3","4","5"]
    pretty = {
        "1": "1 = Strongly disagree",
        "2": "2 = Disagree",
        "3": "3 = Indifferent",
        "4": "4 = Agree",
        "5": "5 = Strongly agree",
    }
    idx = st.radio(title, list(range(len(opts))), format_func=lambda i: pretty.get(opts[i], opts[i]), horizontal=True)
    return opts[idx]

# ---------- discover node names (underscore / space tolerant) ----------
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

# ---------- medians (use FRIENDLY LABELS, not codes) ----------
# You can tune these if you have empirical medians by segment.
empathy_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 3, "50-65": 4},
    "Marital_Status": {"Married": 4, "Single": 4, "Prefer not to say": 3},
    "Shopping_frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 4, "5-6x/week": 4, "6-7x/week": 4},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 4, "Unemployed": 4},
    "Level_of_Education": {"No formal": 3, "Basic": 3, "Diploma": 4, "Degree": 4, "Postgrad": 4},
}
convenience_medians = {
    "Gender": {"Male": 4, "Female": 4, "Prefer not to say": 3},
    "Age": {"18-22": 3, "23-28": 4, "29-35": 4, "35-49": 4, "50-65": 4},
    "Marital_Status": {"Married": 4, "Single": 4, "Prefer not to say": 5},
    "Shopping_frequency": {"1-2x/week": 4, "2-3x/week": 4, "3-4x/week": 5, "5-6x/week": 5, "6-7x/week": 2},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 4, "Unemployed": 4},
    "Level_of_Education": {"No formal": 3, "Basic": 3, "Diploma": 4, "Degree": 4, "Postgrad": 4},
}
customer_trust_medians = {
    "Gender": {"Male": 4, "Female": 3, "Prefer not to say": 3},
    "Age": {"18-22": 4, "23-28": 4, "29-35": 3, "35-49": 4, "50-65": 3},
    "Marital_Status": {"Married": 4, "Single": 3, "Prefer not to say": 3},
    "Shopping_frequency": {"1-2x/week": 3, "2-3x/week": 4, "3-4x/week": 3, "5-6x/week": 5, "6-7x/week": 3},
    "Regular_Customer": {"Regular": 4, "Only when\nneeded": 3, "Only when needed": 3},
    "Employment_Status": {"Employed": 3, "Unemployed": 4},
    "Level_of_Education": {"No formal": 3, "Basic": 3, "Diploma": 4, "Degree": 4, "Postgrad": 4},
}

# =========================================================
# 1) DEMOGRAPHICS
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("1) Choose the customers demographics")
col1, col2 = st.columns(2, gap="large")
with col1:
    ui_gender_code = radio_mapped("Gender", Gender, horizontal=True)
    ui_age_code = radio_mapped("Age", Age, horizontal=True)
    ui_marital_code = radio_mapped("Marital Status", Marital_Status, horizontal=True)
    ui_regular_code = radio_mapped("Customer Type", Regular_Customer, horizontal=True)
with col2:
    ui_empstat_code = radio_mapped("Employment Status", Employment_Status, horizontal=True)
    ui_edu_code = radio_mapped("Level of Education", Level_of_Education, horizontal=True)
    ui_shopfreq_code = radio_mapped("Shopping frequency", Shopping_frequency, horizontal=False)
    
st.markdown('</div>', unsafe_allow_html=True)

# convert codes -> labels for median lookup
def label_of(node_key, code):
    return LABELS.get(node_key, {}).get(code, code)

demo_labels = {
    "Gender": label_of(Gender, ui_gender_code) if ui_gender_code else None,
    "Age": label_of(Age, ui_age_code) if ui_age_code else None,
    "Marital_Status": label_of(Marital_Status, ui_marital_code) if ui_marital_code else None,
    "Employment_Status": label_of(Employment_Status, ui_empstat_code) if ui_empstat_code else None,
    "Level_of_Education": label_of(Level_of_Education, ui_edu_code) if ui_edu_code else None,
    "Shopping_frequency": label_of(Shopping_frequency, ui_shopfreq_code) if ui_shopfreq_code else None,
    "Regular_Customer": label_of(Regular_Customer, ui_regular_code) if ui_regular_code else None,
}

# =========================================================
# 2) AUTO-COMPUTED
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("2) Predetermined purchase intention scores based on demographics")
emp_score = averaged_score_for_var(empathy_medians, demo_labels)
conv_score = averaged_score_for_var(convenience_medians, demo_labels)
trust_score = averaged_score_for_var(customer_trust_medians, demo_labels)

c1, c2, c3 = st.columns(3)
for col, title, val in zip((c1, c2, c3),
                           ("Empathy (1-5)", "Convenience (1-5)", "Customer Trust (1-5)"),
                           (emp_score, conv_score, trust_score)):
    with col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(title, val)
        st.markdown("</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 3) CUSTOMER ANSWERS
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("3) Input your customers' answers in here based on their level of agreement")
q_val_code = likert_radio("Perceived Value", Perceived_Value)
q_qual_code = likert_radio("Perceived Product Quality", Perceived_Product_Quality)
q_price_code = likert_radio("Price Sensitivity", Price_Sensitivity)
q_env_code = likert_radio("Physical Environment", Physical_Environment)

st.markdown('</div>', unsafe_allow_html=True)

# build evidence (send STRING CODES ONLY)
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

# =========================================================
# 4) PREDICTION
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("4) Prediction")

if st.button("Predict Purchase Intention", type="primary"):
    try:
        tgt, pred_class, conf, cls, prob_vec = predict_purchase(bundle, evidence)

        # Pretty labels for classes (customize if you want)
        def pretty_class(c: str) -> str:
            mapping = {"1": "Very Low", "2": "Low", "3": "Medium", "4": "High", "5": "Very High"}
            return mapping.get(str(c), str(c))

        st.success(f"Predicted **{tgt}**: **{pretty_class(pred_class)}**  |  Confidence: **{conf*100:.1f}%**")
        st.progress(int(round(conf*100)))

        prob_df = pd.DataFrame({"Class": [pretty_class(c) for c in cls], "Probability": prob_vec}).sort_values("Probability", ascending=False)
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}), use_container_width=True, hide_index=True)

                # ---------- Assisted selling (optional, does NOT affect BN) ----------
        if conf >= ASSIST_THRESHOLD:
            st.markdown("### 5) Optional: Product interest (for quick assistance)")
            st.caption("Since purchase intention is high, you can capture interest to alert a specialist.")

            # Non-compulsory product choice
           chosen_category = st.selectbox("Which product category...", PRODUCT_CATEGORIES)
            

            # Recommend a staff member
            recommended = suggest_staff(chosen_category)
            st.info(f"Suggested staff member for **{chosen_category}**: **{recommended}**")

            # (Optional) Provide a tiny handoff note you can copy
            handoff = {
                "category": chosen_category,
                "recommended_staff": recommended,
                "predicted_intention_class": str(pred_class),
                "confidence": f"{conf*100:.1f}%"
            }
            with st.expander("Copy handoff details"):
                st.json(handoff)
        else:
            st.markdown("### 5) Optional: Product interest")
            st.caption("Prediction confidence is below 60%. Skipping assisted handoff to focus on higher-likelihood customers.")



    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Evidence (debug)"):
            st.json(evidence)

st.markdown('</div>', unsafe_allow_html=True)
