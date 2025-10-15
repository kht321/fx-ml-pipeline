import os, csv, pickle
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from docker.types import Mount

BASE = "/opt/airflow"
GOLD = f"{BASE}/data/gold"
MODELS = f"{BASE}/models"
# Host paths must be set in the scheduler container env (.env consumed by compose)
HOST_DATA_DIR = os.environ["HOST_DATA_DIR"]
HOST_REPORTS_DIR = os.environ["HOST_REPORTS_DIR"]

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
    schedule=None,      # manual trigger
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    doc_md="Batch scoring, then generate Evidently report via DockerOperator",
) as dag:

    score_batch = PythonOperator(
        task_id="score_batch",
        python_callable=score,
    )

    # Mount host folders into the Evidently container at /data and /reports,
    # and tell the app to read those via env vars.
    data_mount    = Mount(target="/data",    source=HOST_DATA_DIR,    type="bind")
    reports_mount = Mount(target="/reports", source=HOST_REPORTS_DIR, type="bind")

    evidently_report = DockerOperator(
        task_id="evidently_report",
        image="evidently-monitor:0.6.7",
        command=["python", "/app/generate.py"],
        docker_url="unix://var/run/docker.sock",
        mounts=[data_mount, reports_mount],
        environment={"DATA_DIR": "/data", "REPORTS_DIR": "/reports"},
        mount_tmp_dir=False,
        auto_remove="success",
        network_mode="bridge",
    )

    score_batch >> evidently_report

