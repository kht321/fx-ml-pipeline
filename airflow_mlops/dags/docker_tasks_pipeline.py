from datetime import timedelta
import pendulum
import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount  # 👈 use real Mount objects

default_args = {"retries": 1, "retry_delay": timedelta(seconds=10)}

HOST_DATA = os.environ["HOST_DATA_DIR"]
HOST_MODELS = os.environ["HOST_MODELS_DIR"]

data_mount   = Mount(target="/data",   source=HOST_DATA,   type="bind")
models_mount = Mount(target="/models", source=HOST_MODELS, type="bind")


with DAG(
    dag_id="docker_tasks_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    # schedule="@daily",
    catchup=False,
    default_args=default_args,
    doc_md="ETL→Train using separate task images via DockerOperator",
) as dag:

    ingest = DockerOperator(
        task_id="ingest",
        image="etl-tasks:3.1",
        command=["/app/steps/ingest.py"],
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[data_mount],
        network_mode="bridge",
    )

    validate = DockerOperator(
        task_id="validate",
        image="dq-tasks:3.1",
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[data_mount],
        network_mode="bridge",
    )

    transform = DockerOperator(
        task_id="transform",
        image="etl-tasks:3.1",
        command=["/app/steps/transform.py"],
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[data_mount],
        network_mode="bridge",
    )

    features = DockerOperator(
        task_id="features",
        image="etl-tasks:3.1",
        command=["/app/steps/features.py"],
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[data_mount],
        network_mode="bridge",
    )

    train = DockerOperator(
        task_id="train",
        image="trainer-tasks:3.1",
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[data_mount, models_mount],
        network_mode="bridge",
    )

    ingest >> validate >> transform >> features >> train