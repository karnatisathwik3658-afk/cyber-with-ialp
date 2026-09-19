
import pandas as pd
import joblib
import numpy as np

# File paths
dataset_path = r".\DataSet\kddcup.data.cleaned.txt"
encoder_path = r".\models\ordinal_encoder.pkl"
model_path = r".\models\isolation_forest_model.pkl"

# KDD dataset columns
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

print("Loading model and encoder...")

encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

print("Reading dataset sample...")

data = pd.read_csv(
    dataset_path,
    names=columns,
    nrows=1000
)

# Separate features and labels
X = data.drop(columns=["attack_type"])

# Encode categorical features
X[categorical_columns] = encoder.transform(
    X[categorical_columns]
)

# Convert all values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("Making predictions...")

predictions = model.predict(X)
scores = model.decision_function(X)

normal_count = np.sum(predictions == 1)
anomaly_count = np.sum(predictions == -1)

print("\n========== RESULTS ==========")
print("Total records tested:", len(X))
print("Normal-like records:", normal_count)
print("Anomaly-like records:", anomaly_count)
print("Minimum anomaly score:", scores.min())
print("Maximum anomaly score:", scores.max())
print("=============================")

print("\nISOLATION FOREST TEST COMPLETED")