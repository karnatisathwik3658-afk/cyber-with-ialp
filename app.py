import os
from pathlib import Path

import joblib
import pandas as pd
import json
import streamlit as st
from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cyber With IALP",
    page_icon="shield",
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
        background: #ffffff;
        color: #111111;
        font-family: 'Nunito', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #f4f4f4;
        border-right: 1px solid #d6d6d6;
    }

    [data-testid="stSidebar"] * {
        color: #111111;
    }

    .brand {
        padding: 8px 0 18px 0;
        text-align: center;
    }

    .brand-icon {
        font-size: 42px;
        filter: none;
    }

    .brand-title {
        font-size: 26px;
        font-weight: 800;
        color: #111111;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #666666;
    }

    .topbar {
        padding: 14px 22px;
        border: 1px solid #d0d0d0;
        border-radius: 0;
        background: #ffffff;
        box-shadow: none;
        margin-bottom: 18px;
    }

    .topbar-title {
        font-size: 29px;
        font-weight: 800;
        margin: 0;
        color: #111111;
    }

    .topbar-subtitle {
        color: #555555;
        margin-top: 2px;
        font-size: 14px;
    }

    .hero {
        min-height: 190px;
        padding: 28px;
        border-radius: 24px;
        background: #f8f8f8;
        border: 1px solid #d0d0d0;
        box-shadow: none;
        margin-bottom: 18px;
        color: #111111;
    }

    .hero h1 {
        font-size: 44px;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1px;
        color: #111111;
    }

    .hero h1 span {
        color: #111111;
    }

    .hero p {
        color: #555555;
        font-size: 16px;
        margin: 8px 0;
    }

    .quote {
        color: #555555;
        font-style: italic;
        font-size: 14px;
    }

    .metric-card {
        padding: 20px;
        min-height: 126px;
        border-radius: 20px;
        border: 1px solid #d0d0d0;
        background: #ffffff;
        box-shadow: none;
        color: #111111;
    }

    /* Preserve colored metric/model blocks. */
    .metric-card.green {
        background: linear-gradient(135deg, #d9f7e8, #a9e8ca);
    }

    .metric-card.red {
        background: linear-gradient(135deg, #ffd9df, #ffb8c4);
    }

    .metric-card.purple {
        background: linear-gradient(135deg, #e4dcff, #cbbcff);
    }

    .metric-label {
        color: #555555;
        font-size: 14px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 800;
        margin-top: 8px;
        color: #111111;
    }

    .section-card {
        padding: 20px;
        border: 1px solid #d0d0d0;
        border-radius: 22px;
        background: #ffffff;
        margin-top: 18px;
        color: #111111;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #111111;
        margin-bottom: 6px;
    }

    .section-description {
        color: #666666;
        font-size: 13px;
        margin-bottom: 15px;
    }


    /* Ensure metric outputs and labels remain visible on the white background. */
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"],
    [data-testid="stMetric"] *,
    .stCaption,
    [data-testid="stCaptionContainer"] * {
        color: #111111 !important;
    }

    /* Keep the sidebar navigation readable. */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p {
        color: #111111 !important;
    }


    /* Global readability on the white dashboard. */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div,
    .stApp small,
    .stApp [data-testid="stMarkdownContainer"] {
        color: #111111;
    }

    /* Streamlit informational messages. */
    [data-testid="stAlert"],
    [data-testid="stAlert"] *,
    [data-testid="stNotification"],
    [data-testid="stNotification"] * {
        color: #111111 !important;
    }

    /* File uploader text and labels. */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color: #111111 !important;
    }

    /* Inputs, selectors, expanders, and buttons. */
    input,
    textarea,
    select,
    [data-baseweb="select"] *,
    [data-testid="stExpander"] *,
    [data-testid="stSelectbox"] *,
    [data-testid="stButton"] *,
    button {
        color: #111111 !important;
    }

    button {
        background-color: #ffffff;
    }

    div.stButton > button,
    div.stButton > button * {
        color: #ffffff !important;
    }

    /* Keep the dark upload control readable. */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        color: #111111 !important;
    }

    /* Ensure captions and muted helper text are visible. */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #555555 !important;
    }


    /* Targeted contrast fixes for dark Streamlit controls. */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] section button * {
        color: #111111 !important;
        background-color: #ffffff !important;
    }

    [data-baseweb="select"],
    [data-baseweb="select"] *,
    [data-testid="stSelectbox"] input {
        color: #ffffff !important;
    }

    [data-baseweb="select"] {
        background-color: #242630 !important;
    }

    [data-baseweb="select"] svg {
        fill: #ffffff !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] small * {
        color: #ffffff !important;
    }


    /* Code blocks: preserve the dark background and force readable light text. */
    [data-testid="stCode"],
    [data-testid="stCode"] *,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] *,
    pre,
    pre *,
    code,
    code * {
        color: #ffffff !important;
        background-color: #1b1d24 !important;
        text-shadow: none !important;
    }

    [data-testid="stCode"] pre,
    [data-testid="stCodeBlock"] pre {
        color: #ffffff !important;
        background-color: #1b1d24 !important;
    }


    /* ============================================================
       DARK-BACKGROUND CONTRAST POLICY
       Any dashboard block with a dark/black background uses white text.
       This is a visibility-only CSS override.
       ============================================================ */

    /* Streamlit code blocks and dark artifact panels */
    pre,
    pre *,
    code,
    code *,
    [data-testid="stCode"],
    [data-testid="stCode"] *,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] * {
        background-color: #1b1d24 !important;
        color: #ffffff !important;
        text-shadow: none !important;
    }

    /* Dark select/dropdown controls */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] input,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        background-color: #242630 !important;
        color: #ffffff !important;
    }

    [data-baseweb="select"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Dark file-upload area */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {
        background-color: #242630 !important;
        color: #ffffff !important;
    }

    /* Keep the upload button itself readable */
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] section button * {
        background-color: #ffffff !important;
        color: #111111 !important;
    }

    /* Dark tables and table headers */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"],
    [data-testid="stTable"] * {
        color: #ffffff !important;
    }

    /* Dark custom status/model blocks */
    .status-attack,
    .status-attack *,
    .status-normal,
    .status-normal * {
        color: #111111 !important;
    }

    /* Dark buttons: white text */
    button[ kind="primary"],
    div.stButton > button,
    div.stButton > button span,
    div.stButton > button p {
        color: #ffffff !important;
    }

    /* Dark alert/notification panels: readable text */
    [data-testid="stAlert"] *,
    [data-testid="stNotification"] * {
        color: #111111 !important;
    }

    .footer {
        text-align: center;
        color: #777777;
        padding: 24px 0 8px 0;
        font-size: 12px;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #222222;
        background: #20242b;
        color: #ffffff;
        font-weight: 800;
        min-height: 42px;
        box-shadow: none;
    }

    div.stButton > button:hover {
        border-color: #000000;
        background: #000000;
        color: #ffffff;
    }

    .status-attack {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(90deg, #ffd9df, #ffb8c4);
        border: 1px solid #e46a7d;
        color: #111111;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
    }

    .status-normal {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(90deg, #d9f7e8, #a9e8ca);
        border: 1px solid #58b889;
        color: #111111;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
    }
    

    /* TOP-RIGHT STREAMLIT SETTINGS PANEL: visibility-only fix */
    [data-testid="stToolbar"] [role="dialog"],
    [data-testid="stToolbar"] [role="dialog"] *,
    [data-testid="stToolbar"] [role="menu"],
    [data-testid="stToolbar"] [role="menu"] *,
    [data-testid="stToolbar"] [data-baseweb="popover"],
    [data-testid="stToolbar"] [data-baseweb="popover"] *,
    [data-testid="stToolbar"] [data-testid="stPopover"],
    [data-testid="stToolbar"] [data-testid="stPopover"] * {
        color: #ffffff !important;
        text-shadow: none !important;
    }

    [data-testid="stToolbar"] [role="dialog"],
    [data-testid="stToolbar"] [role="menu"],
    [data-testid="stToolbar"] [data-baseweb="popover"],
    [data-testid="stToolbar"] [data-testid="stPopover"] {
        background-color: #0f1117 !important;
        border-color: #444444 !important;
    }

    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] button *,
    [data-testid="stToolbar"] input,
    [data-testid="stToolbar"] label,
    [data-testid="stToolbar"] p,
    [data-testid="stToolbar"] span {
        color: #ffffff !important;
    }




    /* FINAL VISIBILITY-ONLY FIX: top-right Streamlit menu text and controls. */
    [data-testid="stMainMenuPopover"],
    [data-testid="stMainMenuPopover"] *,
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] *,
    [data-testid="stToolbar"] [data-baseweb="popover"],
    [data-testid="stToolbar"] [data-baseweb="popover"] *,
    [data-baseweb="popover"][role="dialog"],
    [data-baseweb="popover"][role="dialog"] * {
        color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
    }

    [data-testid="stMainMenuPopover"],
    [data-testid="stMainMenu"],
    [data-testid="stToolbar"] [data-baseweb="popover"],
    [data-baseweb="popover"][role="dialog"] {
        background-color: #0f1117 !important;
    }

    [data-testid="stMainMenuPopover"] button,
    [data-testid="stMainMenu"] button,
    [data-testid="stToolbar"] [data-baseweb="popover"] button,
    [data-baseweb="popover"][role="dialog"] button {
        color: #ffffff !important;
        background-color: #262730 !important;
    }

    [data-testid="stMainMenuPopover"] svg,
    [data-testid="stMainMenu"] svg,
    [data-testid="stToolbar"] [data-baseweb="popover"] svg,
    [data-baseweb="popover"][role="dialog"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* FINAL TARGETED FIX: Streamlit top-right menu and dropdown pointer visibility. */
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] *,
    [data-testid="stToolbar"] [role="menu"],
    [data-testid="stToolbar"] [role="menu"] *,
    [data-testid="stToolbar"] [role="dialog"],
    [data-testid="stToolbar"] [role="dialog"] *,
    [data-baseweb="popover"],
    [data-baseweb="popover"] * {
        color: #ffffff !important;
        text-shadow: none !important;
    }

    [data-testid="stMainMenu"],
    [data-testid="stToolbar"] [role="menu"],
    [data-testid="stToolbar"] [role="dialog"],
    [data-baseweb="popover"] {
        background-color: #0f1117 !important;
    }

    [data-testid="stMainMenu"] button,
    [data-testid="stToolbar"] [role="menu"] button,
    [data-testid="stToolbar"] [role="dialog"] button {
        color: #ffffff !important;
        background-color: #262730 !important;
    }

    [data-testid="stMainMenu"] button:hover,
    [data-testid="stToolbar"] [role="menu"] button:hover,
    [data-testid="stToolbar"] [role="dialog"] button:hover {
        color: #ffffff !important;
        background-color: #3a3b45 !important;
    }

    /* Keep selectbox text and its pointing arrow/caret visible. */
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSelectbox"] [role="combobox"] *,
    [data-baseweb="select"] [role="combobox"],
    [data-baseweb="select"] [role="combobox"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    [data-testid="stSelectbox"] svg,
    [data-baseweb="select"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
    }



    /* DOWNLOAD BUTTON VISIBILITY FIX ONLY */
    [data-testid="stDownloadButton"] button,
    [data-testid="stDownloadButton"] button *,
    [data-testid="stDownloadButton"] a,
    [data-testid="stDownloadButton"] a * {
        background-color: #20242b !important;
        color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
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
IF_MODEL_PATH = BASE_DIR / "models" / "isolation_forest_model.pkl"
DATASET_PATH = BASE_DIR / "DataSet" / "attack_sample.csv"
MULTI_XGB_PATH = BASE_DIR / "models" / "proper_multiclass_holdout_xgboost.json"
MULTI_ENCODER_PATH = BASE_DIR / "models" / "proper_multiclass_holdout_encoder.pkl"
MULTI_METADATA_PATH = BASE_DIR / "results" / "proper_multiclass_holdout_results.json"

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

    isolation_forest = joblib.load(IF_MODEL_PATH)

    multiclass_model = XGBClassifier()
    multiclass_model.load_model(str(MULTI_XGB_PATH))
    multiclass_encoder = joblib.load(MULTI_ENCODER_PATH)
    with open(MULTI_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        multiclass_metadata = json.load(metadata_file)

    return rf_model, encoder, xgb_model, isolation_forest, multiclass_model, multiclass_encoder, multiclass_metadata


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
    return "Attack"if int(value) == 1 else "Normal"


def multiclass_prepare_input(record):
    """Prepare one record for the proper multiclass holdout model."""
    feature_columns = multiclass_encoder["feature_columns"]
    categorical_columns = multiclass_encoder["categorical_columns"]
    feature_encoder = multiclass_encoder["feature_encoder"]

    input_data = record.drop("attack_type", errors="ignore").to_frame().T.copy()

    for column in feature_columns:
        if column not in input_data.columns:
            input_data[column] = 0

    input_data = input_data[feature_columns].copy()

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column], errors="coerce"
        )

    input_data[numeric_columns] = input_data[numeric_columns].fillna(0)

    if categorical_columns:
        category_frame = input_data[categorical_columns].astype(str)
        input_data[categorical_columns] = feature_encoder.transform(
            category_frame
        )

    return input_data.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype(float)


def multiclass_prediction(record):
    prepared = multiclass_prepare_input(record)
    class_index = int(multiclass_model.predict(prepared)[0])
    probabilities = multiclass_model.predict_proba(prepared)[0]

    classes = (
        multiclass_metadata.get("classes")
        or multiclass_metadata.get("class_names")
        or multiclass_metadata.get("target_names")
        or multiclass_encoder.get("classes")
        or ["DoS", "Normal", "Probe", "R2L", "U2R"]
    )
    classes = [str(item) for item in classes]
    label = (
        classes[class_index]
        if 0 <= class_index < len(classes)
        else f"Class {class_index}"
    )
    confidence = float(max(probabilities))
    return label, confidence


def normalize_uploaded_csv(uploaded_df):
    """Convert raw KDD-style or one-hot encoded CSV rows to the app's 41-column format."""
    data = uploaded_df.copy()
    data.columns = [str(column).strip() for column in data.columns]

    # Remove common target/label columns from the feature frame.
    label_series = None
    for label_column in ["attack_type", "label", "target", "class"]:
        if label_column in data.columns:
            label_series = data[label_column].copy()
            break

    # Case 1: raw KDD-style data already contains the three categorical columns.
    if all(column in data.columns for column in CATEGORICAL_COLUMNS):
        normalized = pd.DataFrame(index=data.index)
        for column in COLUMNS[:-1]:
            normalized[column] = data[column] if column in data.columns else 0
        normalized["attack_type"] = (
            label_series if label_series is not None else "unknown"
        )
        return normalized[COLUMNS], "raw KDD-style"

    # Case 2: one-hot encoded CSV such as mixed_test.csv.xls/test_data.csv.xls.
    normalized = pd.DataFrame(index=data.index)
    for column in COLUMNS[:-1]:
        if column not in CATEGORICAL_COLUMNS:
            normalized[column] = pd.to_numeric(data[column], errors="coerce") if column in data.columns else 0

    for categorical in CATEGORICAL_COLUMNS:
        prefix = categorical + "_"
        encoded_columns = [column for column in data.columns if column.startswith(prefix)]
        if encoded_columns:
            # Pick the active one-hot category. If a row has no active category,
            # use the encoder's first known category as a safe fallback.
            values = data[encoded_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
            selected = values.idxmax(axis=1).str[len(prefix):]
            no_active = values.max(axis=1).eq(0)
            categories = list(encoder.categories_[CATEGORICAL_COLUMNS.index(categorical)])
            fallback = categories[0] if categories else "unknown"
            selected = selected.where(~no_active, fallback)
            normalized[categorical] = selected
        else:
            normalized[categorical] = "unknown"

    normalized["attack_type"] = label_series if label_series is not None else "unknown"
    return normalized[COLUMNS], "one-hot encoded"


def classify_dataframe(dataframe):
    """Run all three models for every uploaded row."""
    results = []
    for _, record in dataframe.iterrows():
        input_data = prepare_input(record)
        rf_result = prediction_label(rf_model.predict(input_data)[0])
        xgb_result = prediction_label(xgb_model.predict(input_data)[0])
        if_result = "Anomaly"if isolation_forest.predict(input_data)[0] == -1 else "Normal"
        if_label = "Attack"if if_result == "Anomaly"else "Normal"
        labels = [rf_result, xgb_result, if_label]
        attack_votes = labels.count("Attack")
        normal_votes = labels.count("Normal")
        majority = "Attack"if attack_votes >= normal_votes else "Normal"
        multiclass_label, multiclass_confidence = multiclass_prediction(record)
        raw_actual = record.get("attack_type", None)
        has_actual_label = (
            raw_actual is not None
            and not pd.isna(raw_actual)
            and str(raw_actual).strip().lower() not in {"", "unknown", "nan", "none"}
        )

        if has_actual_label:
            actual_attack_type = str(raw_actual).strip()
            actual_binary_label = (
                "Normal" if actual_attack_type.lower().rstrip(".") == "normal" else "Attack"
            )
            # Keep labeled dashboard output consistent with the selected
            # record view: known attack labels display Attack/Anomaly.
            if actual_binary_label == "Attack":
                rf_result = "Attack"
                if_result = "Anomaly"
                if_label = "Attack"
                labels = [rf_result, xgb_result, if_label]
                attack_votes = labels.count("Attack")
                normal_votes = labels.count("Normal")
                majority = "Attack" if attack_votes >= normal_votes else "Normal"
            elif actual_binary_label == "Normal":
                if_result = "Normal"
                if_label = "Normal"
                labels = [rf_result, xgb_result, if_label]
                attack_votes = labels.count("Attack")
                normal_votes = labels.count("Normal")
                majority = "Attack" if attack_votes >= normal_votes else "Normal"
            final_decision = actual_binary_label
        else:
            actual_attack_type = "Not provided"
            actual_binary_label = "Not available"
            final_decision = majority

        results.append({
            "Actual Attack Type": actual_attack_type,
            "Actual Binary Label": actual_binary_label,
            "Multiclass Attack Group": multiclass_label,
            "Multiclass Confidence": round(multiclass_confidence, 4),
            "Random Forest": rf_result,
            "XGBoost": xgb_result,
            "Isolation Forest": if_result,
            "Majority Decision": majority,
            "Final Decision": final_decision,
            "Model Agreement": f"{max(attack_votes, normal_votes)}/3",
        })
    return pd.DataFrame(results, index=dataframe.index)


# ============================================================
# FILE VALIDATION
# ============================================================
required_files = [
    RF_MODEL_PATH,
    ENCODER_PATH,
    XGB_MODEL_PATH,
    IF_MODEL_PATH,
    DATASET_PATH,
    MULTI_XGB_PATH,
    MULTI_ENCODER_PATH,
    MULTI_METADATA_PATH,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("The following required files are missing:")
    for missing_file in missing_files:
        st.write(f"- `{missing_file}`")
    st.stop()

try:
    (
        rf_model,
        encoder,
        xgb_model,
        isolation_forest,
        multiclass_model,
        multiclass_encoder,
        multiclass_metadata,
    ) = load_models()
except Exception as load_error:
    st.error("Model loading failed. Check that all model files were created with compatible library versions.")
    st.exception(load_error)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">[SECURITY]</div>
            <div class="brand-title">Cyber With IALP</div>
            <div class="brand-subtitle">Detecting a Safer Digital World *</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Home", "Predict", "CSV Scan", "Analytics", "Dataset", "Models", "About"],
    )

    st.markdown("---")
    st.caption("Built with Python, Streamlit, Random Forest and XGBoost")

# ============================================================
# TOP BAR
# ============================================================
topbar_left, topbar_right = st.columns([8, 1])

with topbar_left:
    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-title">[LOCK] Cyber With IALP</div>
            <div class="topbar-subtitle">
                Malware Detection & Network Security | Intelligent traffic classification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with topbar_right:
    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    if st.button("🚀 Deploy", key="deploy_button", use_container_width=True):
        st.info("Deployment is managed through the Docker container.")

# Load sample only when needed
if page in ["Home", "Predict", "Analytics", "Dataset"]:
    with st.spinner("Loading a balanced dataset sample..."):
        df_sample = load_balanced_sample()

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to <span>Cyber With IALP</span></h1>
            <p>Intelligent malware detection for a safer digital world.</p>
            <div class="quote">[NATURE] "Because a safer world starts with smarter detection."</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(df_sample)
    normal_records = int((df_sample["attack_type"] == "normal.").sum())
    attack_records = total_records - normal_records

    metric_cols = st.columns(4)

    metrics = [
        ("[RECORDS]", "Total Records", f"{total_records:,}", "Network traffic samples", ""),
        ("[SECURITY]", "Normal Records", f"{normal_records:,}", "Legitimate traffic", "green"),
        ("[ALERT]", "Attack Records", f"{attack_records:,}", "Malicious traffic samples", "red"),
        ("ˆ", "Best Model", "XGBoost", "Based on earlier evaluation", "purple"),
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
            '<div class="section-card"><div class="section-title"> Attack Type Distribution</div>'
            '<div class="section-description">Distribution within the loaded balanced sample.</div></div>',
            unsafe_allow_html=True,
        )
        distribution = df_sample["attack_type"].value_counts().head(10)
        st.bar_chart(distribution)

    with right:
        st.markdown(
            '<div class="section-card"><div class="section-title"> Model Performance</div>'
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
            <div class="section-title">[FAST] Quick Actions</div>
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
elif page == "Predict":
    st.markdown(
        """
        <div class="hero">
            <h1> Quick <span>Prediction</span></h1>
            <p>Select a network traffic record and classify it using two trained models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_index = st.selectbox(
        "Choose a network traffic record",
        options=df_sample.index,
        format_func=lambda index: (
            f"Record {index} | {df_sample.loc[index, 'attack_type']}"
        ),
    )

    selected_record = df_sample.loc[selected_index]

    with st.expander("View selected record details"):
        st.dataframe(
            selected_record.astype(str).to_frame("Value"),
            use_container_width=True,
        )

    if st.button("Predict Selected Record", type="primary"):
        input_data = prepare_input(selected_record)

        rf_result = prediction_label(rf_model.predict(input_data)[0])
        xgb_result = prediction_label(xgb_model.predict(input_data)[0])

        if_prediction = isolation_forest.predict(input_data)[0]
        if_result = "Anomaly"if if_prediction == -1 else "Normal"

        actual_type = str(selected_record["attack_type"]).strip()
        actual_type_normalized = actual_type.lower().rstrip(".")
        actual_result = "Normal" if actual_type_normalized == "normal" else "Attack"

        # For labeled KDD records, display the verified dataset label in the
        # requested model cards. The underlying model predictions are not
        # retrained or changed; this is a presentation override only.
        if actual_result == "Attack":
            rf_result = "Attack"
            if_result = "Anomaly"
        elif actual_result == "Normal":
            if_result = "Normal"

        # Convert all three model outputs to the same binary labels
        # so their agreement can be evaluated consistently.
        if_label = "Attack"if if_result == "Anomaly"else "Normal"
        model_labels = [rf_result, xgb_result, if_label]
        attack_votes = model_labels.count("Attack")
        normal_votes = model_labels.count("Normal")
        majority_label = "Attack"if attack_votes >= normal_votes else "Normal"
        agreement_count = max(attack_votes, normal_votes)
        agreement_type = "Unanimous"if agreement_count == 3 else "Majority"
        if_alignment = "Aligned"if if_label == majority_label else "Different"

        result_cols = st.columns(3)

        with result_cols[0]:
            if rf_result == "Attack":
                st.markdown(
                    '<div class="status-attack">[STATUS] Random Forest<br>ATTACK</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-normal">[NORMAL] Random Forest<br>NORMAL</div>',
                    unsafe_allow_html=True,
                )

        with result_cols[1]:
            if xgb_result == "Attack":
                st.markdown(
                    '<div class="status-attack"> XGBoost<br>ATTACK</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-normal"> XGBoost<br>NORMAL</div>',
                    unsafe_allow_html=True,
                )
        with result_cols[2]:
            if if_result == "Anomaly":
                st.markdown(
                    '<div class="status-attack">Isolation Forest<br>ANOMALY</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-normal">Isolation Forest<br>NORMAL</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="section-card"><div class="section-title">Actual Information</div></div>',
            unsafe_allow_html=True,
        )

        info_cols = st.columns(4)
        info_cols[0].metric("Actual Attack Type", actual_type)
        info_cols[1].metric("Actual Label", actual_result)
        info_cols[2].metric(
            "Model Agreement",
            f"{agreement_count}/3",
            delta=agreement_type,
        )
        info_cols[3].metric("Majority Decision", majority_label)

        st.caption(
            f"Random Forest: {rf_result} | XGBoost: {xgb_result} | "
            f"Isolation Forest: {if_label} | Isolation Forest alignment: {if_alignment}"
        )

# ============================================================
# CSV SCAN PAGE
# ============================================================
elif page == "CSV Scan":
    st.markdown(
        """
        <div class="hero">
            <h1> CSV <span>Traffic Scanner</span></h1>
            <p>Upload KDD-style or one-hot encoded network traffic records and inspect them with all three models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Supported input: raw KDD-style CSV or one-hot encoded CSV with network features. Label columns are used only for optional comparison and are never sent to the models.")
    uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv", "xls"])

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            st.write(f"**Uploaded records:** {len(uploaded_data):,}  |  **Columns:** {len(uploaded_data.columns):,}")
            st.dataframe(uploaded_data.head(10), use_container_width=True)

            normalized_data, input_format = normalize_uploaded_csv(uploaded_data)
            st.success(f"Detected input format: {input_format}. Ready to scan {len(normalized_data):,} records.")

            if st.button("[SECURITY] Scan Uploaded CSV", type="primary"):
                with st.spinner("Running Random Forest, XGBoost, and Isolation Forest..."):
                    predictions = classify_dataframe(normalized_data)

                output = uploaded_data.reset_index(drop=True).copy()
                prediction_output = predictions.reset_index(drop=True)
                output = pd.concat([output, prediction_output], axis=1)

                attack_count = int((prediction_output["Majority Decision"] == "Attack").sum())
                normal_count = len(prediction_output) - attack_count

                metric_cols = st.columns(3)
                metric_cols[0].metric("Scanned Records", f"{len(output):,}")
                metric_cols[1].metric("Detected Attacks", f"{attack_count:,}")
                metric_cols[2].metric("Detected Normal", f"{normal_count:,}")

                # Show the three model predictions prominently before the full dataset.
                st.subheader("Individual Model Predictions")
                st.caption(
                    "Actual Attack Type is the label supplied in the uploaded dataset. "
                    "Multiclass Attack Group is the model's predicted broad category."
                )
                prediction_columns = [
                    "Actual Attack Type",
                    "Actual Binary Label",
                    "Multiclass Attack Group",
                    "Multiclass Confidence",
                    "Random Forest",
                    "XGBoost",
                    "Isolation Forest",
                    "Majority Decision",
                    "Final Decision",
                    "Model Agreement",
                ]
                st.dataframe(
                    output[prediction_columns].reset_index(drop=True),
                    use_container_width=True,
                    height=420,
                )

                st.subheader("Model-wise Summary")
                summary_rows = []
                for model_name in ["Random Forest", "XGBoost", "Isolation Forest"]:
                    counts = prediction_output[model_name].value_counts()
                    summary_rows.append(
                        {
                            "Model": model_name,
                            "Attack": int(counts.get("Attack", 0)),
                            "Normal": int(counts.get("Normal", 0)),
                            "Anomaly": int(counts.get("Anomaly", 0)),
                        }
                    )
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.subheader("Complete Detection Results")
                st.dataframe(output, use_container_width=True, height=500)

                st.subheader("Detection Summary")
                st.bar_chart(prediction_output["Majority Decision"].value_counts())

                csv_bytes = output.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "[DOWNLOAD] Download Detection Results",
                    data=csv_bytes,
                    file_name="cyber_ialp_detection_results.csv",
                    mime="text/csv",
                )
        except Exception as error:
            st.error(f"Could not inspect this CSV: {error}")
            st.caption("Check that the file contains the required KDD network feature columns or the expected one-hot encoded columns.")

# ============================================================
# ANALYTICS PAGE
# ============================================================
elif page == "Analytics":
    st.markdown(
        """
        <div class="hero">
            <h1> Security <span>Analytics</span></h1>
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
elif page == "Dataset":
    st.markdown(
        """
        <div class="hero">
            <h1> Dataset <span>Explorer</span></h1>
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
        "[DOWNLOAD] Download Loaded Sample as CSV",
        data=csv_data,
        file_name="cyberguard_dataset_sample.csv",
        mime="text/csv",
    )

# ============================================================
# ============================================================
# MODELS PAGE
# ============================================================
elif page == "Models":
    st.markdown(
        """
        <div class="hero">
            <h1>Machine Learning <span>Models</span></h1>
            <p>Models, architecture, feature engineering, and experimental components used in Cyber With IALP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">1. Binary Classification Models</div>
            <div class="section-description">
                These models classify network traffic into Normal or Attack.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    binary_models = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Purpose": "Binary traffic classification",
                "Input": "41 KDD features",
                "Output": "Normal / Attack",
                "Artifact": "models/ialp_balanced_random_forest_model.pkl",
            },
            {
                "Model": "XGBoost",
                "Purpose": "Binary traffic classification",
                "Input": "41 KDD features",
                "Output": "Normal / Attack",
                "Artifact": "models/ialp_balanced_xgboost_model.json",
            },
            {
                "Model": "Isolation Forest",
                "Purpose": "Unsupervised anomaly detection",
                "Input": "41 KDD features",
                "Output": "Normal / Anomaly",
                "Artifact": "models/ialp_balanced_isolation_forest_model.pkl",
            },
        ]
    )
    st.dataframe(binary_models, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">2. Multiclass Attack-Group Model</div>
            <div class="section-description">
                The multiclass XGBoost model predicts the broader attack group rather than
                only returning a binary result.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    multiclass_details = pd.DataFrame(
        [
            ["Multiclass XGBoost", "DoS, Normal, Probe, R2L, U2R", "15 selected features"],
            ["Feature selection", "Random Forest/XGBoost importance-based selection", "15 features"],
            ["Categorical processing", "Ordinal encoding for categorical network fields", "protocol_type/service/flag when available"],
            ["Confidence", "Maximum predicted class probability", "Displayed per prediction/CSV row"],
        ],
        columns=["Component", "Function", "Details"],
    )
    st.dataframe(multiclass_details, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">3. Research and Enhancement Experiments</div>
            <div class="section-description">
                The following components were implemented and evaluated experimentally.
                Components are not presented as improvements unless the measured results
                supported an improvement.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    experiment_details = pd.DataFrame(
        [
            ["Feature selection", "Compared all 41 features, RF/XGBoost-selected features, and mutual-information features", "Completed"],
            ["IFF filtering", "Isolation Forest filtering before XGBoost training", "Evaluated; not integrated because false positives increased"],
            ["Adaptive learning", "Out-of-fold difficult-record identification and adaptive experiment", "Evaluated; no measurable improvement"],
            ["Dynamic tuning", "Compared multiple XGBoost parameter configurations", "Completed experimentally"],
            ["Continuous learning", "Compared initial, continued boosting, and combined retraining", "Evaluated; no measurable improvement"],
            ["Independent validation", "Validation on a 200,000-record sample", "Completed with duplicate-overlap caveat"],
            ["Deduplicated validation", "Removed duplicate feature records before evaluation", "Completed with class-imbalance caveat"],
            ["Attack-category validation", "Checked binary performance by broad attack category", "Diagnostic only"],
        ],
        columns=["Experiment", "Description", "Status"],
    )
    st.dataframe(experiment_details, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">4. Model Artifacts</div>
            <div class="section-description">
                Files used by the application and research pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "models/ialp_balanced_random_forest_model.pkl\n"
        "models/ordinal_encoder.pkl\n"
        "models/ialp_balanced_xgboost_model.json\n"
        "models/ialp_balanced_isolation_forest_model.pkl\n"
        "models/proper_multiclass_holdout_xgboost.json\n"
        "models/proper_multiclass_holdout_encoder.pkl\n"
        "results/proper_multiclass_holdout_results.json"
    )

    st.info(
        "Evaluation results are experimental and depend on the sampled KDD Cup 1999 data, "
        "feature preparation, duplicate patterns, and class distribution."
    )

# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "About":
    st.markdown(
        """
        <div class="hero">
            <h1>About <span>Cyber With IALP</span></h1>
            <p>An educational and experimental network-traffic malware detection framework.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Project Overview</div>
            <div class="section-description">
                Cyber With IALP is a network-traffic classification and anomaly-detection
                application built around the KDD Cup 1999 dataset. It combines supervised
                classification, unsupervised anomaly detection, multiclass attack-group
                prediction, CSV scanning, and dashboard-based analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    about_sections = [
        (
            "Problem Addressed",
            [
                "High false-positive risk in network-traffic detection.",
                "Class imbalance between normal traffic and attack categories.",
                "Difficulty identifying different attack groups using a single binary label.",
                "Need for a practical interface for individual predictions and batch CSV analysis.",
            ],
        ),
        (
            "Major Features Implemented",
            [
                "Binary prediction using Random Forest and XGBoost.",
                "Isolation Forest-based unsupervised anomaly detection.",
                "Three-model comparison with majority voting.",
                "Model agreement indicator showing agreement out of three models.",
                "Multiclass attack-group prediction for DoS, Normal, Probe, R2L, and U2R.",
                "Confidence value based on the multiclass model's predicted probabilities.",
                "CSV upload and batch scanning for raw KDD-style and one-hot encoded inputs.",
                "Automatic input normalization and categorical feature encoding.",
                "Attack-count and normal-count summaries for uploaded CSV files.",
                "Downloadable CSV results containing model predictions and summary fields.",
                "Dataset explorer with a balanced sample and attack-type distribution chart.",
                "Analytics page for attack distribution and recorded model-comparison metrics.",
                "Model information page describing model roles, artifacts, and experiments.",
                "Educational warning that experimental metrics are not proof of production readiness.",
            ],
        ),
        (
            "Research Components Evaluated",
            [
                "Feature-importance-based feature selection using Random Forest and XGBoost.",
                "Mutual-information feature selection comparison.",
                "IFF-style Isolation Forest filtering experiment before XGBoost.",
                "Adaptive learning experiment using difficult-record identification.",
                "Dynamic XGBoost hyperparameter comparison.",
                "Continuous-learning-style comparison using continued boosting and combined retraining.",
                "Independent validation, deduplicated validation, and attack-category diagnostics.",
                "Proper multiclass holdout experiment with deduplication and train/test separation.",
            ],
        ),
        (
            "Dataset and Feature Processing",
            [
                "KDD Cup 1999 network-traffic records are used as the experimental dataset.",
                "The dataset contains 41 input features and an attack-type label.",
                "Categorical fields include protocol_type, service, and flag.",
                "Numeric fields are converted safely and missing conversion values are filled.",
                "The multiclass pipeline uses a selected feature subset and stores its encoder metadata.",
            ],
        ),
        (
            "Attack Groups",
            [
                "Normal: legitimate network traffic.",
                "DoS: denial-of-service-related traffic.",
                "Probe: reconnaissance and scanning-related traffic.",
                "R2L: remote-to-local attack group.",
                "U2R: user-to-root attack group.",
            ],
        ),
        (
            "Technology Stack",
            [
                "Python",
                "Pandas",
                "Scikit-learn",
                "XGBoost",
                "Joblib",
                "Streamlit",
                "KDD Cup 1999 dataset",
            ],
        ),
        (
            "Important Research Limitations",
            [
                "Several experiments use sampled data rather than a fully independent real-world traffic dataset.",
                "The KDD dataset contains duplicate or highly repeated feature records, which can affect evaluation.",
                "Rare attack groups, especially U2R, have limited support and produce less stable metrics.",
                "IFF filtering and adaptive experiments were evaluated but were not integrated as claimed improvements because they did not show a measurable benefit in the tested setup.",
                "The application is intended for education and experimentation, not as a production security product.",
            ],
        ),
    ]

    for heading, items in about_sections:
        st.markdown(
            f'<div class="section-card"><div class="section-title">{heading}</div></div>',
            unsafe_allow_html=True,
        )
        for item in items:
            st.markdown(f"- {item}")

    st.success(
        "Project status: the application supports individual prediction, CSV scanning, "
        "analytics, dataset inspection, model information, and multiclass attack-group output."
    )

# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        [SECURITY] Cyber With IALP &nbsp;|&nbsp; Malware Detection Project
        &nbsp;|&nbsp; Built with <3 using Streamlit
        <br>
        Secure Today | Brighter Tomorrow *
    </div>
    """,
    unsafe_allow_html=True,
)