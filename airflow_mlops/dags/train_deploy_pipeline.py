import os, csv, pickle, random, json
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

BASE = "/opt/airflow"
GOLD = f"{BASE}/data/gold"
MODELS = f"{BASE}/models"

def train(**_):
    """
        •	Reads features from /opt/airflow/data/gold/features.csv.
        •	Trains a dummy model (just averages + probability) and saves candidate.pkl into /opt/airflow/models.
    """
    src = os.path.join(GOLD, "features.csv")
    xs, ys = [], []
    with open(src) as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["feat_mean"])); ys.append(int(row["label"]))
    thr = sum(xs)/max(1,len(xs)); p1 = sum(ys)/max(1,len(ys))
    model = {"threshold": thr, "p1": p1}
    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, "candidate.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("trained candidate:", model)

def evaluate(**_):
    """
        •	Generates a random AUC score (0.6–0.9).
        •	Writes this metric into candidate_metrics.json.
    """
    auc = round(random.uniform(0.6, 0.9), 3)
    with open(os.path.join(MODELS, "candidate_metrics.json"), "w") as f:
        json.dump({"auc": auc}, f)
    print("candidate AUC", auc)

def register(**_):
    """
        •	Reads candidate_metrics.json, and if AUC ≥ 0.7, promotes candidate.pkl to model_green.pkl.
        •	If AUC < 0.7, raises an error to fail the task.
    """
    with open(os.path.join(MODELS, "candidate_metrics.json")) as f:
        auc = json.load(f)["auc"]
    if auc < 0.7:
        raise ValueError(f"AUC {auc} < 0.7, fail gate")
    import shutil
    shutil.copy(os.path.join(MODELS,"candidate.pkl"),
                os.path.join(MODELS,"model_green.pkl"))
    print("registered to green")

def canary(**_):
    """
        •	Writes a traffic-state file that says “90% traffic to blue, 10% to green”.
	    •	This simulates a canary deployment where only some requests go to the new model.
    """
    with open(os.path.join(MODELS,"traffic_state.json"),"w") as f:
        json.dump({"blue":90, "green":10}, f)
    print("set canary 10% green")

def promote(**_):
    """
        •	Copies model_green.pkl → model_blue.pkl.
        •	Updates traffic to 100% blue (the “new blue” is what used to be green).
        •	This simulates final promotion after a successful canary.
    """ 
    import shutil
    shutil.copy(os.path.join(MODELS,"model_green.pkl"),
                os.path.join(MODELS,"model_blue.pkl"))
    with open(os.path.join(MODELS,"traffic_state.json"),"w") as f:
        json.dump({"blue":100, "green":0}, f)
    print("promoted green to blue")

default_args = {"retries": 1, "retry_delay": timedelta(seconds=10)}

with DAG(
    dag_id="train_deploy_pipeline",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    # schedule="@daily",
    catchup=False,
) as dag:
    t1 = PythonOperator(task_id="train", python_callable=train)
    t2 = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t3 = PythonOperator(task_id="register_green", python_callable=register)
    t4 = PythonOperator(task_id="canary_10pct", python_callable=canary)
    t5 = PythonOperator(task_id="promote_blue_green", python_callable=promote)
    t1 >> t2 >> t3 >> t4 >> t5