import os, csv, pickle
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

BASE = "/opt/airflow"
GOLD = f"{BASE}/data/gold"
MODELS = f"{BASE}/models"

def load_model(slot="blue"):
    path = os.path.join(MODELS, f"model_{slot}.pkl")
    if not os.path.exists(path):
        for p in ["model_green.pkl", "candidate.pkl"]:
            alt = os.path.join(MODELS, p)
            if os.path.exists(alt):
                path = alt; break
    with open(path,"rb") as f:
        return pickle.load(f)

def score(**_):
    model = load_model("blue")
    xs = []
    with open(os.path.join(GOLD,"features.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append((row["user_id"], float(row["feat_mean"])))
    out = []
    for uid, x in xs:
        p1 = model["p1"] if x >= model["threshold"] else 1 - model["p1"]
        out.append((uid, round(p1,3)))
    os.makedirs(f"{BASE}/data/predictions", exist_ok=True)
    with open(f"{BASE}/data/predictions/batch_scores.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["user_id","score"]); w.writerows(out)
    print("wrote predictions/batch_scores.csv")

with DAG(
    dag_id="batch_inference",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    # schedule="@daily",
    catchup=False,
    default_args={"retries":1, "retry_delay": timedelta(seconds=10)},
) as dag:
    PythonOperator(task_id="score_batch", python_callable=score)