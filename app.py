import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cyber With IALP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS - Ghibli-inspired, vibrant cybersecurity UI
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(54, 115, 255, 0.18), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(221, 89, 255, 0.16), transparent 25%),
            linear-gradient(135deg, #07152f 0%, #0b1e42 45%, #10133b 100%);
        color: #f7fbff;
        font-family: 'Nunito', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07132e 0%, #10265a 55%, #24164f 100%);
        border-right: 1px solid rgba(80, 191, 255, 0.45);
    }

    [data-testid="stSidebar"] * {
        color: #f5f8ff;
    }

    .brand {
        padding: 8px 0 18px 0;
        text-align: center;
    }

    .brand-icon {
        font-size: 42px;
        filter: drop-shadow(0 0 12px #27d7ff);
    }

    .brand-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #2ce4ff, #d18aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #b8d9ff;
    }

    .topbar {
        padding: 14px 22px;
        border: 1px solid rgba(91, 191, 255, 0.40);
        border-radius: 22px;
        background: linear-gradient(90deg, rgba(13, 47, 99, .85), rgba(37, 25, 87, .85));
        box-shadow: 0 0 28px rgba(31, 164, 255, .12);
        margin-bottom: 18px;
    }

    .topbar-title {
        font-size: 29px;
        font-weight: 800;
        margin: 0;
    }

    .topbar-subtitle {
        color: #a9eaff;
        margin-top: 2px;
        font-size: 14px;
    }

    .hero {
        min-height: 190px;
        padding: 28px;
        border-radius: 24px;
        background:
            linear-gradient(90deg, rgba(5, 21, 54, .97), rgba(13, 42, 89, .68)),
            radial-gradient(circle at 85% 35%, rgba(245, 116, 255, .45), transparent 30%);
        border: 1px solid rgba(94, 202, 255, .52);
        box-shadow: 0 0 35px rgba(31, 164, 255, .12);
        margin-bottom: 18px;
    }

    .hero h1 {
        font-size: 44px;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #25d9ff, #ff8fe8, #b69bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #c6e9ff;
        font-size: 16px;
        margin: 8px 0;
    }

    .quote {
        color: #6ef3d1;
        font-style: italic;
        font-size: 14px;
    }

    .metric-card {
        padding: 20px;
        min-height: 126px;
        border-radius: 20px;
        border: 1px solid rgba(107, 207, 255, .42);
        background: linear-gradient(135deg, rgba(14, 61, 125, .9), rgba(26, 27, 83, .9));
        box-shadow: 0 8px 25px rgba(0, 0, 0, .16);
    }

    .metric-card.green {
        background: linear-gradient(135deg, rgba(9, 113, 112, .9), rgba(10, 63, 91, .9));
    }

    .metric-card.red {
        background: linear-gradient(135deg, rgba(145, 26, 85, .92), rgba(75, 24, 87, .92));
    }

    .metric-card.purple {
        background: linear-gradient(135deg, rgba(89, 39, 151, .92), rgba(34, 29, 93, .92));
    }

    .metric-label {
        color: #c3e9ff;
        font-size: 14px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-card {
        padding: 20px;
        border: 1px solid rgba(107, 207, 255, .35);
        border-radius: 22px;
        background: rgba(6, 24, 60, .72);
        margin-top: 18px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #f5fbff;
        margin-bottom: 6px;
    }

    .section-description {
        color: #acd6ef;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #a7c8e8;
        padding: 24px 0 8px 0;
        font-size: 12px;
    }

    div.stButton > button {
        border-radius: 18px;
        border: 1px solid #52dfff;
        background: linear-gradient(90deg, #2563ff, #a63cff, #17cde1);
        color: white;
        font-weight: 800;
        min-height: 45px;
        box-shadow: 0 0 18px rgba(67, 198, 255, .18);
    }

    div.stButton > button:hover {
        border-color: #ffffff;
        box-shadow: 0 0 24px rgba(67, 198, 255, .40);
    }

    .status-attack {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(90deg, rgba(190, 31, 75, .9), rgba(103, 22, 96, .9));
        border: 1px solid #ff6d9c;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
    }

    .status-normal {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(90deg, rgba(0, 137, 119, .9), rgba(14, 85, 121, .9));
        border: 1px solid #62f4cf;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATHS AND DATA DEFINITIONS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
RF_MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "ordinal_encoder.pkl"
XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
DATASET_PATH = BASE_DIR / "DataSet" / "kddcup.data.cleaned.txt"

COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


# ============================================================
# LOADING FUNCTIONS
# ============================================================
@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))

    return rf_model, encoder, xgb_model


@st.cache_data
def load_balanced_sample():
    normal_parts = []
    attack_parts = []
    normal_count = 0
    attack_count = 0

    for chunk in pd.read_csv(
        DATASET_PATH,
        names=COLUMNS,
        chunksize=100000,
        low_memory=False,
    ):
        if normal_count < 2500:
            normal_part = chunk[chunk["attack_type"] == "normal."].head(
                2500 - normal_count
            )
            normal_parts.append(normal_part)
            normal_count += len(normal_part)

        if attack_count < 2500:
            attack_part = chunk[chunk["attack_type"] != "normal."].head(
                2500 - attack_count
            )
            attack_parts.append(attack_part)
            attack_count += len(attack_part)

        if normal_count >= 2500 and attack_count >= 2500:
            break

    normal_data = pd.concat(normal_parts, ignore_index=True)
    attack_data = pd.concat(attack_parts, ignore_index=True)

    data = pd.concat([normal_data, attack_data], ignore_index=True)
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def prepare_input(record):
    input_data = record.drop("attack_type").to_frame().T.copy()

    numeric_columns = [
        column for column in input_data.columns
        if column not in CATEGORICAL_COLUMNS
    ]

    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column],
            errors="coerce",
        )

    input_data[numeric_columns] = input_data[numeric_columns].fillna(0)

    input_data[CATEGORICAL_COLUMNS] = encoder.transform(
        input_data[CATEGORICAL_COLUMNS]
    )

    return input_data.astype(float)


def prediction_label(value):
    return "Attack" if int(value) == 1 else "Normal"


# ============================================================
# FILE VALIDATION
# ============================================================
required_files = [
    RF_MODEL_PATH,
    ENCODER_PATH,
    XGB_MODEL_PATH,
    DATASET_PATH,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("The following required files are missing:")
    for missing_file in missing_files:
        st.write(f"- `{missing_file}`")
    st.stop()

rf_model, encoder, xgb_model = load_models()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div class="brand-title">Cyber With IALP</div>
            <div class="brand-subtitle">Detecting a Safer Digital World ✨</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Predict", "📊 Analytics", "🗃️ Dataset", "🧠 Models", "ℹ️ About"],
    )

    st.markdown("---")
    st.caption("Built with Python, Streamlit, Random Forest and XGBoost")

# ============================================================
# TOP BAR
# ============================================================
st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">🔒 Cyber With IALP</div>
        <div class="topbar-subtitle">
            Malware Detection & Network Security • Intelligent traffic classification
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load sample only when needed
if page in ["🏠 Home", "🔍 Predict", "📊 Analytics", "🗃️ Dataset"]:
    with st.spinner("Loading a balanced dataset sample..."):
        df_sample = load_balanced_sample()

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to <span>Cyber With IALP</span></h1>
            <p>Intelligent malware detection for a safer digital world.</p>
            <div class="quote">🌱 “Because a safer world starts with smarter detection.”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(df_sample)
    normal_records = int((df_sample["attack_type"] == "normal.").sum())
    attack_records = total_records - normal_records

    metric_cols = st.columns(4)

    metrics = [
        ("🗄️", "Total Records", f"{total_records:,}", "Network traffic samples", ""),
        ("🛡️", "Normal Records", f"{normal_records:,}", "Legitimate traffic", "green"),
        ("🚨", "Attack Records", f"{attack_records:,}", "Malicious traffic samples", "red"),
        ("📈", "Best Model", "XGBoost", "Based on earlier evaluation", "purple"),
    ]

    for column, (icon, label, value, description, color) in zip(metric_cols, metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card {color}">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="section-card"><div class="section-title">📊 Attack Type Distribution</div>'
            '<div class="section-description">Distribution within the loaded balanced sample.</div></div>',
            unsafe_allow_html=True,
        )
        distribution = df_sample["attack_type"].value_counts().head(10)
        st.bar_chart(distribution)

    with right:
        st.markdown(
            '<div class="section-card"><div class="section-title">🧠 Model Performance</div>'
            '<div class="section-description">Previously measured evaluation results.</div></div>',
            unsafe_allow_html=True,
        )
        performance = pd.DataFrame(
            {
                "Model": ["Random Forest", "XGBoost"],
                "Accuracy": [99.9855, 99.9878],
                "Precision": [99.9992, 99.9972],
                "Recall": [99.9827, 99.9875],
                "F1 Score": [99.9910, 99.9924],
            }
        ).set_index("Model")
        st.dataframe(performance.style.format("{:.4f}%"), use_container_width=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">⚡ Quick Actions</div>
            <div class="section-description">
                Use the navigation menu to predict traffic, explore the dataset, or review model information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "🔍 Predict":
    st.markdown(
        """
        <div class="hero">
            <h1>🔍 Quick <span>Prediction</span></h1>
            <p>Select a network traffic record and classify it using two trained models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_index = st.selectbox(
        "Choose a network traffic record",
        options=df_sample.index,
        format_func=lambda index: (
            f"Record {index} • {df_sample.loc[index, 'attack_type']}"
        ),
    )

    selected_record = df_sample.loc[selected_index]

    with st.expander("View selected record details"):
        st.dataframe(
            selected_record.astype(str).to_frame("Value"),
            use_container_width=True,
        )

    if st.button("🔍 Predict Selected Record", type="primary"):
        input_data = prepare_input(selected_record)

        rf_result = prediction_label(rf_model.predict(input_data)[0])
        xgb_result = prediction_label(xgb_model.predict(input_data)[0])

        actual_type = selected_record["attack_type"]
        actual_result = "Normal" if actual_type == "normal." else "Attack"

        result_cols = st.columns(2)

        with result_cols[0]:
            if rf_result == "Attack":
                st.markdown(
                    '<div class="status-attack">🌐 Random Forest<br>ATTACK</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-normal">🌿 Random Forest<br>NORMAL</div>',
                    unsafe_allow_html=True,
                )

        with result_cols[1]:
            if xgb_result == "Attack":
                st.markdown(
                    '<div class="status-attack">🤖 XGBoost<br>ATTACK</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-normal">🤖 XGBoost<br>NORMAL</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="section-card"><div class="section-title">📌 Actual Information</div></div>',
            unsafe_allow_html=True,
        )

        info_cols = st.columns(3)
        info_cols[0].metric("Actual Attack Type", actual_type)
        info_cols[1].metric("Actual Label", actual_result)
        info_cols[2].metric(
            "Model Agreement",
            "Yes" if rf_result == xgb_result else "No",
        )

# ============================================================
# ANALYTICS PAGE
# ============================================================
elif page == "📊 Analytics":
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Security <span>Analytics</span></h1>
            <p>Explore attack distribution and previously recorded model performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    attack_distribution = df_sample["attack_type"].value_counts()

    st.markdown(
        '<div class="section-card"><div class="section-title">Attack Type Distribution</div>'
        '<div class="section-description">Top attack categories in the balanced sample.</div></div>',
        unsafe_allow_html=True,
    )
    st.bar_chart(attack_distribution.head(15))

    st.markdown(
        '<div class="section-card"><div class="section-title">Model Comparison</div>'
        '<div class="section-description">Evaluation values from your earlier notebook experiment.</div></div>',
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Random Forest": [99.9855, 99.9992, 99.9827, 99.9910],
            "XGBoost": [99.9878, 99.9972, 99.9875, 99.9924],
        }
    ).set_index("Metric")

    st.bar_chart(comparison)
    st.dataframe(comparison.style.format("{:.4f}%"), use_container_width=True)

# ============================================================
# DATASET PAGE
# ============================================================
elif page == "🗃️ Dataset":
    st.markdown(
        """
        <div class="hero">
            <h1>🗃️ Dataset <span>Explorer</span></h1>
            <p>Inspect a balanced sample of KDD Cup 1999 network traffic records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"Loaded records: **{len(df_sample):,}**")
    st.write(f"Number of features including attack type: **{df_sample.shape[1]}**")
    st.dataframe(df_sample.head(100), use_container_width=True)

    csv_data = df_sample.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Loaded Sample as CSV",
        data=csv_data,
        file_name="cyberguard_dataset_sample.csv",
        mime="text/csv",
    )

# ============================================================
# MODELS PAGE
# ============================================================
elif page == "🧠 Models":
    st.markdown(
        """
        <div class="hero">
            <h1>🧠 Machine Learning <span>Models</span></h1>
            <p>Information about the trained models used by Cyber With IALP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_cols = st.columns(2)

    with model_cols[0]:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🌲 Random Forest</div>
                <div class="section-description">
                    Ensemble learning model that combines multiple decision trees.
                    Useful for robust classification and feature importance analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with model_cols[1]:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">⚡ XGBoost</div>
                <div class="section-description">
                    Gradient boosting model that builds trees sequentially to improve
                    classification performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("Model files:")
    st.code(
        "models/random_forest_model.pkl\n"
        "models/ordinal_encoder.pkl\n"
        "models/xgboost_model.json"
    )

# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "ℹ️ About":
    st.markdown(
        """
        <div class="hero">
            <h1>ℹ️ About <span>Cyber With IALP</span></h1>
            <p>An educational malware detection framework using network traffic data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Project Technology Stack</div>
            <div class="section-description">
                Python • Pandas • Scikit-learn • XGBoost • Streamlit • KDD Cup 1999
            </div>
            <br>
            <div class="section-title">Important Note</div>
            <div class="section-description">
                This project is intended for educational experimentation. The evaluation
                results were obtained using a random train-test split and should not be
                interpreted as proof of production-level cybersecurity performance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        🛡️ Cyber With IALP &nbsp;|&nbsp; Malware Detection Project
        &nbsp;|&nbsp; Built with ❤️ using Streamlit
        <br>
        Secure Today • Brighter Tomorrow ✨
    </div>
    """,
    unsafe_allow_html=True,
)
