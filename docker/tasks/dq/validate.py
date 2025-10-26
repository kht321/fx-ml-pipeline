import os, sys, pandas as pd
DATA = os.environ.get("DATA_DIR","/data")
csv_path = os.path.join(DATA, "bronze", "docker_tasks_events.csv")
if not os.path.exists(csv_path):
    sys.exit(f"Missing input file: {csv_path}")
df = pd.read_csv(csv_path)
if len(df) < 50:
    sys.exit("Too few rows")
if not set(["user_id","feature_a","label"]).issubset(df.columns):
    sys.exit("Missing required columns")
if not df["feature_a"].between(0,10).all():
    sys.exit("feature_a out of range")
if not set(df["label"].unique()).issubset({0,1}):
    sys.exit("invalid label values")
print("DQ passed")
