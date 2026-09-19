
import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. File paths
# ==========================================

DATASET_PATH = r"DataSet\kddcup.data.cleaned.txt"
ENCODER_PATH = r"models\ordinal_encoder.pkl"
MODEL_PATH = r"models\isolation_forest_model.pkl"

# ==========================================
# 2. Dataset column names
# ==========================================

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

categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

# ==========================================
# 3. Load the existing encoder
# ==========================================

print("Loading existing encoder...")

encoder = joblib.load(ENCODER_PATH)

print("Encoder loaded successfully.")

# ==========================================
# 4. Read the dataset in chunks
# ==========================================

print("Reading dataset in chunks...")

chunk_size = 100_000
sample_per_chunk = 2_500

sample_parts = []
total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATASET_PATH,
        names=columns,
        header=None,
        chunksize=chunk_size
    ),
    start=1
):
    total_rows += len(chunk)

    # Randomly sample records from this chunk
    current_sample_size = min(sample_per_chunk, len(chunk))

    sampled_chunk = chunk.sample(
        n=current_sample_size,
        random_state=42 + chunk_number
    )

    sample_parts.append(sampled_chunk)

    if chunk_number % 10 == 0:
        print(
            f"Processed approximately "
            f"{total_rows:,} rows..."
        )

print("Dataset reading completed.")

# ==========================================
# 5. Combine sampled records
# ==========================================

sample_data = pd.concat(
    sample_parts,
    ignore_index=True
)

# Limit to 100,000 records
sample_data = sample_data.sample(
    n=min(100_000, len(sample_data)),
    random_state=42
).reset_index(drop=True)

print("Sample shape:", sample_data.shape)

# ==========================================
# 6. Prepare numeric features
# ==========================================

print("Preparing features...")

X = sample_data.drop(
    columns=["attack_type"]
)

# Encode categorical columns
X[categorical_columns] = encoder.transform(
    X[categorical_columns]
)

# Ensure all values are numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace missing or infinite values
X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

X = X.fillna(0)

print("Feature shape:", X.shape)

# ==========================================
# 7. Train Isolation Forest
# ==========================================

print("Training Isolation Forest...")

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X)

print("Isolation Forest training completed.")

# ==========================================
# 8. Save the model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    isolation_forest,
    MODEL_PATH
)

print("Model saved at:", MODEL_PATH)
print("ISOLATION FOREST TRAINING SUCCESSFUL")