import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from google import genai
from sklearn.linear_model import LogisticRegression
from bias_utils import preprocess_data, prepare_dataset, measure_bias, mitigate_bias, group_outcome_rates, map_group_value
import json

# ─────────────────────────────────────────────
#  Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fairsight • BIAS Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --surface2:  #1a1e2a;
    --border:    #252936;
    --accent:    #5b8dee;
    --accent2:   #e05b7f;
    --success:   #3ecf8e;
    --warn:      #f5a623;
    --danger:    #e05b7f;
    --text:      #e8eaf2;
    --muted:     #7c82a0;
    --radius:    12px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Main container */
.main .block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1200px;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #13161e 0%, #1a1e2a 50%, #13161e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(91,141,238,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(224,91,127,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem;
    background: linear-gradient(90deg, #e8eaf2 30%, #5b8dee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(91,141,238,0.12);
    border: 1px solid rgba(91,141,238,0.3);
    color: #5b8dee;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Card component ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid var(--accent);
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.blue::before  { background: var(--accent); }
.metric-card.pink::before  { background: var(--accent2); }
.metric-card.green::before { background: var(--success); }
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
}
.metric-card.blue  .metric-value { color: var(--accent); }
.metric-card.pink  .metric-value { color: var(--accent2); }
.metric-card.green .metric-value { color: var(--success); }

/* ── Status banners ── */
.status-success {
    background: rgba(62,207,142,0.08);
    border: 1px solid rgba(62,207,142,0.25);
    border-left: 4px solid var(--success);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: #8df5c9;
    font-size: 0.92rem;
    margin: 0.8rem 0;
}
.status-warn {
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.25);
    border-left: 4px solid var(--warn);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: #fbd08a;
    font-size: 0.92rem;
    margin: 0.8rem 0;
}
.status-danger {
    background: rgba(224,91,127,0.08);
    border: 1px solid rgba(224,91,127,0.25);
    border-left: 4px solid var(--danger);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: #f0a0b5;
    font-size: 0.92rem;
    margin: 0.8rem 0;
}
.status-info {
    background: rgba(91,141,238,0.08);
    border: 1px solid rgba(91,141,238,0.25);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: #a0bcf5;
    font-size: 0.92rem;
    margin: 0.8rem 0;
}

/* ── AI explanation box ── */
.ai-box {
    background: linear-gradient(135deg, #13161e, #1a1e2a);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
}
.ai-box::before {
    content: '✦ AI INSIGHT';
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--accent);
    display: block;
    margin-bottom: 0.8rem;
}
.ai-model-tag {
    display: inline-block;
    background: rgba(91,141,238,0.1);
    border: 1px solid rgba(91,141,238,0.2);
    color: var(--muted);
    font-size: 0.7rem;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 20px;
    margin-top: 0.8rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Selectbox & inputs ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.55rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(91,141,238,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.8rem 0 !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Streamlit native overrides ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Syne', sans-serif !important; }

/* ── Caption ── */
.stCaption { color: var(--muted) !important; font-size: 0.78rem !important; }

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    background: var(--accent);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    border-radius: 50%;
    margin-right: 0.4rem;
    flex-shrink: 0;
}

/* Progress bar for DI */
.di-bar-wrap {
    background: var(--surface2);
    border-radius: 100px;
    height: 10px;
    margin: 0.5rem 0 1rem;
    overflow: hidden;
    position: relative;
}
.di-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.di-ideal-marker {
    position: absolute;
    top: -3px;
    width: 2px; height: 16px;
    background: var(--success);
    border-radius: 2px;
}

/* Scrollable dataset preview */
.preview-wrap {
    max-height: 240px;
    overflow-y: auto;
    border-radius: var(--radius);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    client = None

def dataset_to_xy(dataset):
    result = dataset.convert_to_dataframe()
    df = result[0] if isinstance(result, tuple) else result
    label = dataset.label_names[0]
    X = df.drop(columns=[label])
    y = df[label]
    return X, y

def get_ai_explanation(unpriv_label, priv_label, gap, before, after, protected_col, rates_dict, rates_dict_after):
    disadvantaged = unpriv_label
    privileged = priv_label
    assert disadvantaged != privileged
    prompt = f"""
A Fairness analysis was performed.

Protected Attribute : {protected_col},
Disadvantaged group : {disadvantaged},
Privileged group    : {privileged},

Outcome gap              : {gap:.2f}
Disparate impact before  : {before:.2f},
Disparate impact after   : {after:.2f},

Group outcome rates before : {rates_dict},
Group outcome rates after  : {rates_dict_after}

Explain:
1. What this bias means
2. Why it might occur
3. Whether mitigation helped
4. What actions should be taken
"""
    model = "gemini-2.5-pro" if gap > 0.2 else "gemini-2.5-flash"
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text, model
    except Exception:
        return "AI Explanation Unavailable (fallback used)", "fallback"

def validate_target(df, target_col):
    return df[target_col].nunique() == 2

def ensure_binary(df, col):
    if df[col].nunique() == 2:
        return df, col
    if df[col].dtype != 'object':
        df[col] = (df[col] > df[col].median()).astype(int)
    else:
        top = df[col].value_counts().idxmax()
        df[col] = (df[col] == top).astype(int)
    return df, col

def di_bar_html(value, color):
    """Render a visual disparate-impact progress bar clamped to 0–1.5."""
    pct = min(max(float(value) / 1.5, 0), 1) * 100
    ideal_pct = (1.0 / 1.5) * 100
    return f"""
    <div class="di-bar-wrap">
        <div class="di-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
        <div class="di-ideal-marker" style="left:{ideal_pct:.1f}%;"></div>
    </div>
    """

def styled_banner(text, kind="info"):
    st.markdown(f'<div class="status-{kind}">{text}</div>', unsafe_allow_html=True)

def display_group_analysis(rates_dict, rates_dict_after, protected_col,
                           privileged_value, unprivileged_value):
    if len(rates_dict) < 2:
        st.error("Not enough groups to compare bias.")
        st.stop()

    # Show individual rates
    for group, rate in rates_dict.items():
        label = map_group_value(protected_col, group)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"<small style='color:var(--muted)'>{label}</small>", unsafe_allow_html=True)
            bar_pct = int(rate * 100)
            col_color = "#5b8dee" if int(group) == int(privileged_value) else "#e05b7f"
            st.markdown(f"""
            <div style="background:var(--surface2);border-radius:100px;height:8px;margin:4px 0 8px;">
                <div style="width:{bar_pct}%;height:100%;background:{col_color};border-radius:100px;"></div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"<span style='font-family:Syne,sans-serif;font-weight:700;color:{col_color}'>{rate:.2f}</span>",
                        unsafe_allow_html=True)

    # Chart if AFTER exists
    if rates_dict_after and len(rates_dict_after) > 0:
        rates_dict_int  = {int(k): v for k, v in rates_dict.items()}
        rates_dict_after_int = {int(k): v for k, v in rates_dict_after.items()}

        common_groups = sorted(set(rates_dict_int.keys()) & set(rates_dict_after_int.keys()))
        before_vals   = [rates_dict_int[g] for g in common_groups]
        after_vals    = [rates_dict_after_int[g] for g in common_groups]
        labels        = [map_group_value(protected_col, g) for g in common_groups]
        x             = range(len(common_groups))

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("#13161e")
        ax.set_facecolor("#13161e")

        bars_b = ax.bar([i - 0.22 for i in x], before_vals, width=0.40,
                        color="#5b8dee", alpha=0.75, label="Before")
        bars_a = ax.bar([i + 0.22 for i in x], after_vals,  width=0.40,
                        color="#3ecf8e", alpha=0.85, label="After (Reweighted)")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, color="#e8eaf2", fontsize=11)
        ax.set_ylabel("Positive Outcome Rate", color="#7c82a0", fontsize=10)
        ax.set_title("Outcome Rate — Before vs After Reweighing",
                     color="#e8eaf2", fontsize=12, fontweight="bold", pad=14)
        ax.tick_params(colors="#7c82a0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#252936")
        ax.legend(facecolor="#1a1e2a", edgecolor="#252936",
                  labelcolor="#e8eaf2", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, color="#252936", linewidth=0.6)
        ax.set_axisbelow(True)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    data_for_gap = rates_dict_after if rates_dict_after else rates_dict
    data_int = {int(k): v for k, v in data_for_gap.items()}
    priv_rate   = data_int[int(privileged_value)]
    unpriv_rate = data_int[int(unprivileged_value)]
    gap         = priv_rate - unpriv_rate

    priv_label   = map_group_value(protected_col, privileged_value)
    unpriv_label = map_group_value(protected_col, unprivileged_value)

    return gap, unpriv_label, priv_label


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Fairness AI</div>
    <div class="hero-title">⚖️ FairSight</div>
    <p class="hero-sub">Detect, visualize, and mitigate algorithmic bias in your datasets — powered by AIF360 & Gemini AI</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STEP 1 — Upload
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="step-badge">1</span> Upload Dataset</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a CSV or JSON file",
    type=["csv", "json"],
    help="Supported formats: CSV, JSON (flat or nested)"
)

if uploaded_file is None:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2.5rem;">
        <div style="font-size:2.5rem;margin-bottom:0.6rem;">📂</div>
        <div style="color:var(--muted);font-size:0.9rem;">
            Upload a dataset above to begin your fairness analysis.<br>
            <span style="font-size:0.8rem;opacity:0.6;">Common datasets: Adult Income, COMPAS, German Credit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Parse file ──
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".json"):
    try:
        df = pd.read_json(uploaded_file)
    except ValueError:
        data = json.load(uploaded_file)
        df = pd.json_normalize(data)
else:
    st.error("Unsupported file type")
    st.stop()

df = df.copy()


# ─────────────────────────────────────────────
#  STEP 2 — Configure
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="step-badge">2</span> Configure Analysis</div>',
            unsafe_allow_html=True)

with st.container():
    st.markdown(f"""
    <div class="card card-accent">
        <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
            <div>
                <div class="metric-label">Rows</div>
                <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:var(--accent)">{df.shape[0]:,}</div>
            </div>
            <div>
                <div class="metric-label">Columns</div>
                <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:var(--accent2)">{df.shape[1]}</div>
            </div>
            <div>
                <div class="metric-label">File</div>
                <div style="font-size:0.9rem;color:var(--muted)">{uploaded_file.name}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👁 Preview Dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=220)

    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        label_col = st.selectbox(
            "🎯 Target Column (what to predict)",
            df.columns,
            help="The outcome column — must be or become binary."
        )
    with cfg_col2:
        protected_col = st.selectbox(
            "🛡 Protected Attribute",
            df.columns,
            help="e.g. gender, race, age — the attribute to check for bias."
        )

# ─────────────────────────────────────────────
#  RUN BUTTON
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 3])
with run_col:
    if st.button("⚡ Run Bias Analysis", use_container_width=True):
        st.session_state.run_analysis = True


# ─────────────────────────────────────────────
#  ANALYSIS
# ─────────────────────────────────────────────
if not st.session_state.get("run_analysis"):
    st.stop()

st.markdown("---")
st.markdown('<div class="section-header"><span class="step-badge">3</span> Analysis Results</div>',
            unsafe_allow_html=True)

try:
    # ── Validation ──
    if label_col.lower() in ["race", "gender", "sex"]:
        st.error("⚠️ You selected a protected attribute as target. Please choose a different target column.")
        st.stop()

    if df[label_col].dtype != 'object' and df[label_col].nunique() > 10:
        styled_banner("⚠️ Target appears continuous — binary conversion may reduce meaning. Consider a classification column.", "warn")

    if not validate_target(df, label_col):
        styled_banner("Target is not binary — auto-converting using median split.", "warn")
        df, label_col = ensure_binary(df, label_col)
        styled_banner(f"✓ <strong>{label_col}</strong> converted to binary (median-based). Original data meaning may be reduced.", "info")

    if not validate_target(df, protected_col):
        styled_banner("Protected attribute is not binary — auto-converting.", "warn")
        df, protected_col = ensure_binary(df, protected_col)
        styled_banner(f"✓ <strong>{protected_col}</strong> converted to binary (median-based).", "info")

    # ── Preprocess ──
    df = df.copy()
    df = preprocess_data(df, label_col, protected_col)
    processed_values = sorted(df[protected_col].unique())

    if len(processed_values) < 2:
        st.error("Protected attribute must have at least 2 distinct groups.")
        st.stop()

    # ── Group selection ──
    if protected_col.lower() == "race":
        styled_banner("Race mapped to binary (majority group vs others) for fairness evaluation.", "info")
        privileged_value   = 1
        unprivileged_value = 0
    else:
        display_map = {v: map_group_value(protected_col, v) for v in processed_values}
        grp_col1, grp_col2 = st.columns([2, 3])
        with grp_col1:
            selected_display = st.selectbox("👑 Select Privileged Group", list(display_map.values()))
        reverse_map        = {v: k for k, v in display_map.items()}
        privileged_value   = reverse_map[selected_display]
        unprivileged_value = [v for v in processed_values if v != privileged_value][0]

    priv_lbl   = map_group_value(protected_col, privileged_value)
    unpriv_lbl = map_group_value(protected_col, unprivileged_value)
    st.caption(f"Privileged: **{priv_lbl}** · Unprivileged: **{unpriv_lbl}** · Attribute: **{protected_col}**")

    # ── AIF360 pipeline ──
    with st.spinner("Running fairness pipeline…"):
        dataset = prepare_dataset(df, label_col, protected_col)

        X, y   = dataset_to_xy(dataset)
        model_before = LogisticRegression(max_iter=1000)
        model_before.fit(X, y)
        y_pred_before = model_before.predict(X)

        dataset_pred_before        = dataset.copy()
        dataset_pred_before.labels = y_pred_before.reshape(-1, 1)

        before       = measure_bias(dataset, protected_col, privileged_value, unprivileged_value)
        rates        = group_outcome_rates(dataset, protected_col)
        rates_dict   = {k: float(v) for k, v in dict(rates).items()}

        dataset_fixed = mitigate_bias(dataset, protected_col, privileged_value, unprivileged_value)
        X_fixed, y_fixed = dataset_to_xy(dataset_fixed)
        weights = dataset_fixed.instance_weights

        model_after = LogisticRegression(max_iter=1000)
        model_after.fit(X_fixed, y_fixed, sample_weight=weights)
        y_pred_after = model_after.predict(X_fixed)

        dataset_pred_after        = dataset_fixed.copy()
        dataset_pred_after.labels = y_pred_after.reshape(-1, 1)

        before_model_bias = measure_bias(dataset_pred_before, protected_col, privileged_value, unprivileged_value)
        after_model_bias  = measure_bias(dataset_pred_after,  protected_col, privileged_value, unprivileged_value)

        after        = measure_bias(dataset_fixed, protected_col, privileged_value, unprivileged_value)
        rates_after  = group_outcome_rates(dataset_fixed, protected_col, use_weights=True)
        rates_dict_after = {k: float(v) for k, v in dict(rates_after).items()}

    # ── Disparate Impact Summary ──
    st.markdown("### 📊 Disparate Impact")

    improvement = abs(1 - before) - abs(1 - after)
    di_color_before = "#e05b7f" if abs(before - 1) > 0.2 else "#f5a623" if abs(before - 1) > 0.1 else "#3ecf8e"
    di_color_after  = "#e05b7f" if abs(after  - 1) > 0.2 else "#f5a623" if abs(after  - 1) > 0.1 else "#3ecf8e"

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Before DI</div>
            <div class="metric-value" style="color:{di_color_before}">{before:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(di_bar_html(before, di_color_before), unsafe_allow_html=True)
    with dc2:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-label">After DI</div>
            <div class="metric-value" style="color:{di_color_after}">{after:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(di_bar_html(after, di_color_after), unsafe_allow_html=True)
    with dc3:
        imp_color = "#3ecf8e" if improvement > 0 else "#e05b7f"
        imp_sign  = "+" if improvement > 0 else ""
        st.markdown(f"""
        <div class="metric-card pink">
            <div class="metric-label">Improvement</div>
            <div class="metric-value" style="color:{imp_color}">{imp_sign}{improvement:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("🟢 Green marker on bar = ideal DI of 1.0 · Range 0.8–1.25 is generally fair")

    # Model vs Data DI
    with st.expander("🤖 Model-level Bias (Predicted Labels)", expanded=False):
        mc1, mc2 = st.columns(2)
        mc1.metric("Before (Model predictions)", f"{before_model_bias:.3f}")
        mc2.metric("After  (Reweighted model)",  f"{after_model_bias:.3f}")

    # ── Mitigation verdict ──
    st.markdown("### 🛡 Mitigation Result")
    if abs(after - 1.0) < 0.01:
        styled_banner("✅ <strong>Bias completely removed</strong> — Disparate impact is at 1.0", "success")
    elif improvement > 0:
        styled_banner(f"✅ <strong>Bias reduced</strong> — improvement of {improvement:.3f} via Reweighing", "success")
    else:
        styled_banner("❌ <strong>Bias not improved</strong> — consider alternative mitigation strategies", "danger")

    st.markdown("---")

    # ── Group comparison chart ──
    st.markdown("### 📈 Group Outcome Rates")
    gap, unpriv_label, priv_label = display_group_analysis(
        rates_dict,
        rates_dict_after,
        protected_col,
        privileged_value,
        unprivileged_value
    )
    gap_abs = abs(gap)

    st.markdown(f"""
    <div class="card" style="margin-top:0.8rem;">
        <div class="metric-label">Outcome Gap (|Priv − Unpriv|)</div>
        <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
                    color:{'#e05b7f' if gap_abs>0.2 else '#f5a623' if gap_abs>0.1 else '#3ecf8e'}">
            {gap_abs:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if gap_abs > 0.2:
        if gap > 0:
            styled_banner(f"🔴 <strong>HIGH BIAS</strong> — {unpriv_label} group is significantly disadvantaged", "danger")
        else:
            styled_banner(f"🔴 <strong>HIGH BIAS</strong> — {priv_label} group is significantly disadvantaged", "danger")
    elif gap_abs > 0.1:
        if gap > 0:
            styled_banner(f"🟡 <strong>MODERATE BIAS</strong> — {unpriv_label} group shows unequal outcomes", "warn")
        else:
            styled_banner(f"🟡 <strong>MODERATE BIAS</strong> — {priv_label} group shows unequal outcomes", "warn")
    else:
        styled_banner("🟢 <strong>No significant bias detected</strong> — outcome gap is within acceptable range", "success")

    st.markdown("---")

    # ── AI Explanation ──
    st.markdown("### 🧠 AI Explanation")
    with st.spinner("Generating AI explanation…"):
        ai_output, model_used = get_ai_explanation(
            unpriv_label, priv_label, gap,
            before, after, protected_col,
            rates_dict, rates_dict_after
        )

    st.markdown(f"""
    <div class="ai-box">
        {ai_output}
        <div class="ai-model-tag">Generated with {model_used}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Final decision recommendation ──
    st.markdown("### ✅ Decision Recommendation")
    st.markdown("""
    <div style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        Based on post-mitigation Disparate Impact (ideal = 1.0, acceptable range = 0.8–1.25)
    </div>
    """, unsafe_allow_html=True)

    if 0.8 <= after <= 1.25:
        styled_banner("✅ <strong>FAIR</strong> — Model is within acceptable fairness range. Safe to deploy with monitoring.", "success")
    elif 0.6 <= after < 0.8:
        styled_banner("⚠️ <strong>CAUTION</strong> — Moderate bias remains. Review sensitive use-cases before deployment.", "warn")
    else:
        styled_banner("❌ <strong>HIGH RISK</strong> — Significant bias persists. Not recommended for deployment without further mitigation.", "danger")

    # Root cause note
    st.markdown(f"""
    <div class="card" style="margin-top:1rem;">
        <div style="font-size:0.8rem;color:var(--muted);margin-bottom:0.4rem;">📌 ROOT CAUSE NOTE</div>
        <div style="font-size:0.9rem;color:var(--text)">
            <strong>{protected_col}</strong> may be correlated with other features that influence predictions.
            This bias originates from patterns in historical training data — not necessarily intentional discrimination.
        </div>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Analysis error: {e}")
    st.exception(e)