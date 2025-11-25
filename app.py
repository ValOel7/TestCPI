# app.py — Streamlit BN inference: Purchase Intention + inferred Products

import sys, platform, pickle, json, os
import numpy as np
import pandas as pd
import streamlit as st

# ===== Config =====
ASSIST_THRESHOLD = 0.60   # show staff suggestion if PI confidence >= 60%
TOP_K = 3                 # show top-k product categories
SHOW_STAFF_SUGGESTION = True

# Optional: product -> roster (used only if SHOW_STAFF_SUGGESTION=True)
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
    roster = STAFF_ROSTER.get(category, [])
    if not roster:
        return "Any available associate"
    return roster[hash(category) % len(roster)]

# ---- Session defaults ----
if "pred" not in st.session_state:
    st.session_state.pred = None          # (target, class_str)
    st.session_state.conf = 0.0           # float 0..1
    st.session_state.classes = []         # class order for PI
    st.session_state.prob_vec = None      # np.array for PI distribution
    st.session_state.top_products = None  # df with top-k products

# ---- Import VE only (we load a fitted model from pickle) ----
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

# IMPORTANT: use the patched pickle that contains Product_Category
PICKLE_PATH = "bn_pgmpy_products.pkl"

# ---------- Page + styles ----------
st.set_page_config(page_title="Purchase Intention Real-Time Assistant", page_icon="🛍️", layout="centered")
st.markdown(
    """
    <style>
      :root {
        --accent: #2563eb;
        --bg-section: #fbfcfe;
        --bg-card: #f8f9fb;
        --bd-soft: #eef0f4;
        --text: #0b1220;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --bg-section: #0f1420;
          --bg-card: #0b1220;
          --bd-soft: #1e293b;
          --text: #e5e7eb;
        }
      }
      .block-container {max-width: 980px;}
      body, .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5 { color: var(--text) !important; }
      h1, .stMarkdown h1 { letter-spacing: -0.3px; }
      h2, .stMarkdown h2 { margin-top: 0.25rem; }

      .section {
        background: var(--bg-section);
        border:1px solid var(--bd-soft);
        border-radius:14px; padding:18px 18px 12px; margin: 14px 0 10px;
      }
      .qcard {
        background: var(--bg-section);
        border:1px solid var(--bd-soft);
        border-radius:14px; padding:16px; margin:10px 0;
        box-shadow: 0 1px 0 rgba(10,31,68,0.03);
      }
      .qtitle { font-weight:600; margin-bottom:8px; }
      .badge {
        display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
        background:#eef2ff; color:#3730a3; border:1px solid #e0e7ff; margin-left:8px;
      }
      @media (prefers-color-scheme: dark) {
        .badge { background:#1e293b; color:#bfdbfe; border-color:#334155; }
      }
      .metric-card {
        background: var(--bg-card);
        padding:12px 16px; border-radius:14px; border:1px solid var(--bd-soft);
      }
      .stRadio > label {font-weight:600;}
      .st-emotion-cache-16idsys p { margin-bottom: 0.35rem; } /* tighten labels */
      input[type="radio"]:checked { accent-color: var(--accent); }

      /* Mobile radio spacing */
      @media (max-width: 420px) {
        .stRadio [role="radiogroup"] { gap: 4px !important; }
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛍️ Purchase Intention Real-Time Assistant")

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

for k in ("model", "target", "classes", "state_names"):
    if k not in bundle:
        st.error(f"❌ Pickle missing required key: {k}")
        st.stop()

model = bundle["model"]
TARGET = bundle["target"]
classes = bundle["classes"]
state_names = bundle["state_names"]  # dict node -> list of states

# pgmpy compatibility shims
if not hasattr(model, "check_model"):
    def _noop_check_model(*args, **kwargs): return True
    try: model.check_model = _noop_check_model
    except Exception: pass

if not hasattr(model, "factors"):
    try: cpds = model.get_cpds()
    except Exception: cpds = []
    factors = []
    for cpd in cpds:
        tf = getattr(cpd, "to_factor", None)
        if callable(tf):
            try: factors.append(tf()); continue
            except Exception: pass
        factors.append(cpd)
    try: model.factors = factors
    except Exception: pass

# ---------- helpers ----------
def pick_existing_node(state_names_dict, candidates):
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

def predict_both(bundle_obj, evidence: dict):
    ve = VariableElimination(bundle_obj["model"])

    # Purchase Intention
    q_pi = ve.query([bundle_obj["target"]], evidence=evidence, show_progress=False)
    pi_states = q_pi.state_names[bundle_obj["target"]]
    pi_probs  = q_pi.values.flatten().astype(float)
    pi_idx    = int(pi_probs.argmax())
    pi_pred   = pi_states[pi_idx]
    pi_conf   = float(pi_probs[pi_idx])

    # Products (if present)
    products_topk = products_all = None
    if "Product_Category" in bundle_obj["model"].nodes():
        q_pc = ve.query(["Product_Category"], evidence=evidence, show_progress=False)
        pc_states = q_pc.state_names["Product_Category"]
        pc_probs  = q_pc.values.flatten().astype(float)
        products_all  = pd.DataFrame({"Product": pc_states, "Probability": pc_probs}).sort_values("Probability", ascending=False)
        products_topk = products_all.head(TOP_K)

    return (pi_pred, pi_conf, pi_states, pi_probs, products_topk, products_all)

# ---------- label mapping (pretty UI ↔ codes the model expects) ----------
LABEL_MAP_PATH = "state_label_map.json"  # optional override
DEFAULT_LABELS = {
    "Gender": {"1": "Male", "2": "Female", "3": "Prefer not to say"},
    "Age": {"1": "18-22", "2": "23-28", "3": "29-35", "4": "35-49", "5": "50-65"},
    "Marital_Status": {"1": "Married", "2": "Single", "3": "Prefer not to say"},
    "Employment_Status": {"1": "Employed", "2": "Unemployed", 1: "Employed", 2: "Unemployed"},
    "Level_of_Education": {
        "1": "No formal", "2": "Basic", "3": "Diploma", "4": "Degree", "5": "Postgrad",
        1: "No formal", 2: "Basic", 3: "Diploma", 4: "Degree", 5: "Postgrad",
    },
    "Shopping_frequency": {"1": "1-2x/week", "2": "2-3x/week", "3": "3-4x/week", "4": "5-6x/week", "5": "6-7x/week"},
    "Regular_Customer": {"1": "Regular", "2": "Only when needed"},
}
def load_label_map(state_names_dict) -> dict:
    labels = {}
    if isinstance(state_names_dict, dict):
        for node, states in state_names_dict.items():
            base = DEFAULT_LABELS.get(node, {})
            labels[node] = {s: base.get(s, s) for s in states}
    if os.path.exists(LABEL_MAP_PATH):
        try:
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                override = json.load(f)
            for node, mapping in override.items():
                if node in labels:
                    labels[node].update(mapping)
        except Exception:
            pass
    return labels

LABELS = load_label_map(state_names)

def radio_mapped(title: str, node_key: str, *, horizontal: bool = True):
    opts = state_names.get(node_key, []) if isinstance(state_names, dict) else []
    if not opts:
        return None
    pretty = [LABELS.get(node_key, {}).get(code, code) for code in opts]
    idx = st.radio(title, list(range(len(opts))), format_func=lambda i: pretty[i], horizontal=horizontal)
    return opts[idx]

def likert_radio(title: str, node_key: str):
    opts = state_names.get(node_key, ["1","2","3","4","5"]) if isinstance(state_names, dict) else ["1","2","3","4","5"]
    pretty = {
        "1": "1 = Strongly disagree",
        "2": "2 = Disagree",
        "3": "3 = Indifferent",
        "4": "4 = Agree",
        "5": "5 = Strongly agree",
    }
    idx = st.radio(title, list(range(len(opts))), format_func=lambda i: pretty.get(opts[i], opts[i]), horizontal=True)
    return opts[idx]

# ---------- discover node names ----------
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

# ---------- medians (friendly labels) ----------
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
st.markdown('<div class="section qcard"><div class="qtitle">1) Choose the customer\'s demographics <span class="badge">Step 1</span></div>', unsafe_allow_html=True)
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

# convert codes -> labels for the medians
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
# 2) AUTO-COMPUTED (from demographics)
# =========================================================
st.markdown('<div class="section qcard"><div class="qtitle">2) Predetermined scores from demographics <span class="badge">Step 2</span></div>', unsafe_allow_html=True)
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
# 3) CUSTOMER ANSWERS (Likert)
# =========================================================
st.markdown('<div class="section qcard"><div class="qtitle">3) Customer answers (Likert) <span class="badge">Step 3</span></div>', unsafe_allow_html=True)
q_val_code  = likert_radio("Do you think this store provides good value for money (affordable for what you get)?", Perceived_Value)
q_qual_code = likert_radio("Do you think the overall quality of the products in this shop is good?", Perceived_Product_Quality)
q_price_code= likert_radio("Would you still shop here if prices increased slightly (e.g., for convenience/proximity)?", Price_Sensitivity)
q_env_code  = likert_radio("Is the store appealing, clean, tidy, and a pleasant shopping environment?", Physical_Environment)
st.markdown('</div>', unsafe_allow_html=True)

# Build evidence (send STRING CODES ONLY)
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
# 4) PREDICTION (PI + Products)
# =========================================================
st.markdown('<div class="section qcard"><div class="qtitle">4) Prediction <span class="badge">Step 4</span></div>', unsafe_allow_html=True)

if st.button("Predict Purchase Intention", type="primary"):
    try:
        pi_pred, pi_conf, pi_states, pi_probs, products_topk, products_all = predict_both(bundle, evidence)

        st.session_state.pred = (TARGET, pi_pred)
        st.session_state.conf = pi_conf
        st.session_state.classes = pi_states
        st.session_state.prob_vec = pi_probs
        st.session_state.top_products = products_topk  # <- always set, even if None

        st.rerun()
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if st.session_state.pred is not None:
    tgt, pred_class = st.session_state.pred
    conf = st.session_state.conf
    cls_states = st.session_state.classes
    pi_vec = st.session_state.prob_vec
    top_products = st.session_state.get("top_products")

    # Pretty mapping for PI
    mapping = {"1":"Very Low","2":"Low","3":"Medium","4":"High","5":"Very High"}
    pretty = mapping.get(str(pred_class), str(pred_class))

    st.success(f"Predicted **{tgt}**: **{pretty}**  |  Confidence: **{conf*100:.1f}%**")
    st.progress(int(round(conf*100)))

    # PI distribution (top-3 only)
    pi_df = pd.DataFrame({"Class": [mapping.get(str(c), str(c)) for c in cls_states],
                          "Probability": pi_vec})
    pi_df = pi_df.sort_values("Probability", ascending=False).head(3)
    st.dataframe(pi_df.style.format({"Probability": "{:.3f}"}), use_container_width=True, hide_index=True)

    # Products Top-K
    if top_products is not None and len(top_products) > 0:
        st.subheader("Top product categories (inferred)")
        st.dataframe(
            top_products.style.format({"Probability": "{:.3f}"}),
            use_container_width=True, hide_index=True
        )

        if SHOW_STAFF_SUGGESTION and conf >= ASSIST_THRESHOLD:
            top_product = str(top_products.iloc[0]["Product"])
            staff = suggest_staff(top_product)
            st.info(f"Please call **{staff}** to assist with **{top_product}**.")

st.markdown('</div>', unsafe_allow_html=True)
