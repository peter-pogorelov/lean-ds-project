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


def transform_dataset(processed_dataset_path: str):
    """
    Apply dataset transformations for specific model

    :param processed_dataset_path: path to the processed dataset
    :type processed_dataset_path: str
    :return: transformed dataset of any type
    """    
    with open(processed_dataset_path, 'r') as f:
        return f.read() + ' transformed '


def transform_local(config_path: str):
    """
    Run data transformation phase on the processed dataset, that is stored on local filesystem. 
    The source dataset is assumed to be stored in /data/interim, and the transformed datased is to be
    stored in /data/processed folder.

    :param config_path: path to configuration file
    :type config_path: str
    """    
    with ContextualizedDirectory() as directory:
        tuning_dataset_path = directory.interim.joinpath('dataset_tune.csv')
        validation_dataset_path = directory.interim.joinpath('dataset_validate.csv')

        tuning_dataset_path_transformed = directory.processed.joinpath('dataset_tune.bin')
        validation_dataset_path_transformed = directory.processed.joinpath('dataset_validate.bin')
        
        tuning_transformed = transform_dataset(tuning_dataset_path)
        validation_transformed = transform_dataset(validation_dataset_path)

        joblib.dump(tuning_transformed, tuning_dataset_path_transformed)
        joblib.dump(validation_transformed, validation_dataset_path_transformed)


def transform_mlflow(config_path: str, run_id: str):
    """
    Run transformation phase on the dataset, that is stored in MLFlow. All the intermediate results
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
        tuning_dataset_path = directory.interim.joinpath('dataset_tune.csv')
        validation_dataset_path = directory.interim.joinpath('dataset_validate.csv')

        tuning_dataset_path_transformed = directory.processed.joinpath('dataset_tune.bin')
        validation_dataset_path_transformed = directory.processed.joinpath('dataset_validate.bin')

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id) as run:
            client.download_artifacts(run.info.run_id, 'interim/dataset_tune.csv', directory.data)
            client.download_artifacts(run.info.run_id, 'interim/dataset_validate.csv', directory.data)

            tuning_transformed = transform_dataset(tuning_dataset_path)
            validation_transformed = transform_dataset(validation_dataset_path)

            joblib.dump(tuning_transformed, tuning_dataset_path_transformed)
            joblib.dump(validation_transformed, validation_dataset_path_transformed)

            mlflow.log_artifact(tuning_dataset_path_transformed, 'processed')
            mlflow.log_artifact(validation_dataset_path_transformed, 'processed')


@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    transform_local(config)
    #transform_mlflow(config, MLFLOW_RUN_ID)
                

if __name__ == '__main__':
    main()
