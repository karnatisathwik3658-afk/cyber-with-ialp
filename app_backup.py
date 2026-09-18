
import streamlit as st
import joblib
import pandas as pd
import os
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Malware Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Malware Detection System")
st.write("Network traffic classification using Random Forest and XGBoost")

# File paths
RF_MODEL_PATH = "models/random_forest_model.pkl"
ENCODER_PATH = "models/ordinal_encoder.pkl"
XGB_MODEL_PATH = "models/xgboost_model.json"
DATASET_PATH = "DataSet/kddcup.data.cleaned.txt"

# Dataset column names
columns = [
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
    "dst_host_srv_rerror_rate", "attack_type"
]

categorical_columns = ["protocol_type", "service", "flag"]


# Load models
@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    xgb_model = XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)

    return rf_model, encoder, xgb_model


# Load a small dataset sample

@st.cache_data
def load_dataset_sample():

    normal_records = []
    attack_records = []

    # Read the dataset in manageable chunks
    for chunk in pd.read_csv(
        DATASET_PATH,
        names=columns,
        chunksize=100000
    ):

        normal_chunk = chunk[
            chunk["attack_type"] == "normal."
        ]

        attack_chunk = chunk[
            chunk["attack_type"] != "normal."
        ]

        normal_records.append(normal_chunk.head(2500))
        attack_records.append(attack_chunk.head(2500))

        # Stop when both categories have enough records
        normal_count = sum(len(data) for data in normal_records)
        attack_count = sum(len(data) for data in attack_records)

        if normal_count >= 2500 and attack_count >= 2500:
            break

    normal_data = pd.concat(normal_records).head(2500)
    attack_data = pd.concat(attack_records).head(2500)

    # Combine normal and attack records
    data = pd.concat(
        [normal_data, attack_data],
        ignore_index=True
    )

    # Shuffle records
    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return data

# Check required files
required_files = [
    RF_MODEL_PATH,
    ENCODER_PATH,
    XGB_MODEL_PATH,
    DATASET_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()


# Load everything
rf_model, encoder, xgb_model = load_models()
df_sample = load_dataset_sample()

st.success("Models and dataset loaded successfully!")

st.subheader("Select a Network Traffic Record")

selected_index = st.selectbox(
    "Choose a record:",
    df_sample.index,
    format_func=lambda index:
        f"Record {index} - {df_sample.loc[index, 'attack_type']}"
)

selected_record = df_sample.loc[selected_index]

st.write("### Selected Record")
st.dataframe(
    selected_record.astype(str).to_frame("Value")
)


if st.button("🔍 Predict", type="primary"):

    # Remove actual attack label
    input_data = selected_record.drop("attack_type").to_frame().T

    # Define numeric columns
    numeric_columns = [
        column for column in input_data.columns
        if column not in categorical_columns
    ]

    # Convert numeric columns to numbers
    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column],
            errors="coerce"
        )

    # Replace missing numeric values with 0
    input_data[numeric_columns] = (
        input_data[numeric_columns].fillna(0)
    )

    # Encode categorical columns
    input_data[categorical_columns] = encoder.transform(
        input_data[categorical_columns]
    )

    # Ensure all features are numeric
    input_data = input_data.astype(float)

    # Make predictions
    rf_prediction = rf_model.predict(input_data)[0]
    xgb_prediction = xgb_model.predict(input_data)[0]

    rf_result = "Attack" if rf_prediction == 1 else "Normal"
    xgb_result = "Attack" if xgb_prediction == 1 else "Normal"

    actual_type = selected_record["attack_type"]
    actual_label = (
        "Normal" if actual_type == "normal." else "Attack"
    )

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Random Forest", rf_result)

    with col2:
        st.metric("XGBoost", xgb_result)

    st.subheader("Actual Information")

    st.write(f"**Actual Attack Type:** {actual_type}")
    st.write(f"**Actual Label:** {actual_label}")