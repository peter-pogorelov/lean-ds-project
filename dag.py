import pendulum
from airflow.decorators import dag, task

PYTHON_VERSION='3.9.13'#'3.7.13'
CONFIG_PATH='/opt/airflow/projects/${CI_PROJECT_NAME}/config.yml'
REQUIREMENTS_PATH='/opt/airflow/projects/${CI_PROJECT_NAME}/requirements.txt'
REQUIREMENTS_CONTENT = ''
with open(REQUIREMENTS_PATH, 'r') as f:
    lines = f.readlines()
    REQUIREMENTS_CONTENT = '\n'.join([line for line in lines if not line.startswith('git')])

@dag(
    schedule_interval=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['etl', 'retrain', 'mlflow'],
)
def lean_ds_project_mlflow():
    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def create_run(config_path: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.lifecycle import create_run

        return create_run(config_path=config_path)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def download_data(config_path: str, mlflow_run_id: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.data.download import download_mlflow

        download_mlflow(config_path=config_path, run_id=mlflow_run_id)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def process(config_path: str, mlflow_run_id: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.features.process import process_mlflow

        process_mlflow(config_path=config_path, run_id=mlflow_run_id)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def transform(config_path: str, mlflow_run_id: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.features.transform import transform_mlflow

        transform_mlflow(config_path=config_path, run_id=mlflow_run_id)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def split(config_path: str, mlflow_run_id: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.features.split import split_mlflow

        split_mlflow(config_path=config_path, run_id=mlflow_run_id)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def train(config_path: str, mlflow_run_id: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.models.train import train_mlflow

        train_mlflow(config_path=config_path, run_id=mlflow_run_id)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def validate(config_path: str, mlflow_run_id: str):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.models.validate import validate_mlflow

        return validate_mlflow(config_path=config_path, run_id=mlflow_run_id)

    @task.virtualenv(requirements=REQUIREMENTS_CONTENT, system_site_packages=False, python_version=PYTHON_VERSION, queue='cpu')
    def submit_metrics(config_path: str, metrics: dict):
        import sys
        sys.path.append('/opt/airflow/dags')
        from lean_ds_project_mlflow.lifecycle.submit_metrics import submit_metrics

        submit_metrics(config_path=config_path, metrics=metrics)
    
    run_id = create_run(CONFIG_PATH)
    
    metrics = download_data(CONFIG_PATH, run_id) >> process(CONFIG_PATH, run_id) >> split(CONFIG_PATH, run_id) \
        >> transform(CONFIG_PATH, run_id) >> train(CONFIG_PATH, run_id) >> validate(CONFIG_PATH, run_id)

    submit_metrics(CONFIG_PATH, metrics)


lean_ds_project_mlflow = lean_ds_project_mlflow()