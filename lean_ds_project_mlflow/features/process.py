import os
import yaml
import click
import joblib
import mlflow

from loguru import logger
from mlflow.tracking import MlflowClient
from lean_ds_project_mlflow import ContextualizedDirectory

#UNCOMMENT TO HAVE MLFLOW ACCESS FROM LOCAL MACHINE
#os.environ['MLFLOW_TRACKING_URI'] = 'http://10.80.20.26:5000'
#MLFLOW_RUN_ID is the existing run
#MLFLOW_RUN_ID = 'bfedefdd118c4959aab41c7cb296378e'


def process_dataset(raw_dataset_path: str, processed_dataset_path: str):
    """
    Preprocessing phase (primarily, the cleaning phase and adaptation of raw data for the experiments). 
    The data that comes from this phase can be transformed in a sense, that is is ready to be used with different
    models.

    :param raw_dataset_path: path to raw dataset
    :type raw_dataset_path: str
    :param processed_dataset_path: path to processed dataset to be stored
    :type processed_dataset_path: str
    """    
    with open(raw_dataset_path, 'r') as f_s:
        with open(processed_dataset_path, 'w') as f_d:
            f_d.write(f_s.read())


def process_local(config_path: str):
    """
    Run preprocessing phase on the files, which are stored on local filesystem. The source dataset is assumed
    to be stored in /data/raw, and the preprocessed datased is to be stored in /data/interim folder.

    :param config_path: path to configuration file
    :type config_path: str
    """    
    with ContextualizedDirectory() as directory:
        raw_dataset_path = directory.raw.joinpath('dataset.csv')
        processed_dataset_path = directory.interim.joinpath('dataset.csv')
        process_dataset(raw_dataset_path, processed_dataset_path)


def process_mlflow(config_path: str, run_id: str):
    """
    Run preprocessing phase on the files, which are stored in MLFlow. All the intermediate results
    are to be stored in MLFlow.

    :param config_path: path to configuration file
    :type config_path: str
    :param run_id: id of run
    :type run_id: str
    """        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        experiment_name = config['experiment']

    client = MlflowClient()

    with ContextualizedDirectory() as directory:
        raw_dataset_path = directory.raw.joinpath('dataset.csv')
        processed_dataset_path = directory.interim.joinpath('dataset.csv')
        
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id) as run:
            client.download_artifacts(run.info.run_id, 'raw/dataset.csv', directory.data)
            process_dataset(raw_dataset_path, processed_dataset_path)
            mlflow.log_artifact(processed_dataset_path, 'interim')


@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    process_local(config)
    #process_mlflow(config, MLFLOW_RUN_ID)
           

if __name__ == '__main__':
    main()
