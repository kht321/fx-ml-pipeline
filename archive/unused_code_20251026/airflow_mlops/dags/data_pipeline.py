# ./dags/data_pipeline.py
import os, csv, random
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

# Paths inside the Airflow container (mapped to your host by docker-compose)
BASE = "/opt/airflow"
# BASE = "/tmp/airflow-demo"  # TEMP for diagnosis
BRONZE = f"{BASE}/data/bronze"
SILVER = f"{BASE}/data/silver"
GOLD   = f"{BASE}/data/gold"

def prepare_dirs():
    print("[prepare_dirs] uid/gid:", os.getuid(), os.getgid())
    print("[prepare_dirs] cwd:", os.getcwd())
    for p in (BRONZE, SILVER, GOLD, f"{BASE}/logs"):
        os.makedirs(p, exist_ok=True)
        try:
            os.chmod(p, 0o777)  # dev/demo only
        except Exception as e:
            print("[prepare_dirs] chmod skipped:", repr(e))
        test = os.path.join(p, "_write_test")
        with open(test, "w") as f:
            f.write("ok ...")
        print("[prepare_dirs] ensured & wrote:", test)
    print("[prepare_dirs] done")

def ingest():
    os.makedirs(BRONZE, exist_ok=True)
    path = os.path.join(BRONZE, "events.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "feature_a", "label"])
        for i in range(100):  # dummy data
            w.writerow([i, random.randint(0, 10), random.randint(0, 1)])
    print(f"[ingest] wrote {path}")

def validate():
    path = os.path.join(BRONZE, "events.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing bronze file: {path}")
    # simple sanity checks
    with open(path) as f:
        n = sum(1 for _ in f)  # includes header
    if n <= 1:
        raise ValueError("Bronze file is empty.")
    print(f"[validate] rows including header: {n}")

def transform():
    os.makedirs(SILVER, exist_ok=True)
    src = os.path.join(BRONZE, "events.csv")
    dst = os.path.join(SILVER, "clean.csv")

    kept = 0
    with open(src) as fin, open(dst, "w", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            # simple rule: keep only records with feature_a >= 3
            if int(row["feature_a"]) >= 3:
                w.writerow(row)
                kept += 1
    print(f"[transform] kept {kept} rows -> {dst}")

def build_features():
    os.makedirs(GOLD, exist_ok=True)
    src = os.path.join(SILVER, "clean.csv")
    dst = os.path.join(GOLD, "features.csv")

    agg = {}
    with open(src) as f:
        r = csv.DictReader(f)
        for row in r:
            uid = row["user_id"]
            agg.setdefault(uid, {"sum": 0, "cnt": 0, "label": row["label"]})
            agg[uid]["sum"] += int(row["feature_a"])
            agg[uid]["cnt"] += 1

    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "feat_mean", "label"])
        for uid, a in agg.items():
            mean = a["sum"] / max(1, a["cnt"])
            w.writerow([uid, round(mean, 3), a["label"]])

    print(f"[features] wrote {dst}")

default_args = {
    "owner": "demo",
    "retries": 0,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,              # manual runs only
    catchup=False,
    tags=["demo", "bronze-silver-gold"],
) as dag:
    t0 = PythonOperator(task_id="prepare_dirs", python_callable=prepare_dirs)
    t1 = PythonOperator(task_id="ingest", python_callable=ingest)
    t2 = PythonOperator(task_id="validate", python_callable=validate)
    t3 = PythonOperator(task_id="transform", python_callable=transform)
    t4 = PythonOperator(task_id="build_features", python_callable=build_features)

    t0 >> t1 >> t2 >> t3 >> t4