import os
import yaml
import click
import joblib
import mlflow

from loguru import logger
from mlflow.tracking import MlflowClient
from lean_ds_project_mlflow import ContextualizedDirectory


def split_dataset(processed_dataset_path: str, split_ratio: float, seed: int) -> tuple:
    """
    Split dataset to tuning and validation subsets. tuning dataset is to be used for model selection, hyperparameter tuning
    and model tranining. Validation dataset is reserved for final validation.

    :param processed_dataset_path: path to processed dataset
    :type processed_dataset_path: str
    :param split_ratio: the share of tranining data in dataset
    :type split_ratio: float
    :param seed: fixed random seed
    :type seed: int
    :return: tuning and validation dataset
    :rtype: tuple
    """
    dataset = None
    with open(processed_dataset_path, 'r') as f_s:
        dataset = f_s.read()

    tune_dataset = dataset
    val_dataset = dataset

    return tune_dataset, val_dataset


def split_local(config_path: str):
    """
    Split preprocessed dataset, that is stored on local filesystem.
    The source dataset is assumed to be stored in /data/preprocessed as well as the resulting two files for
    traning and validation.

    :param config_path: path to configuration file
    :type config_path: str
    """   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        seed = config['dataset']['seed']
        validate_ratio = config['dataset']['validate']
    
    with ContextualizedDirectory() as directory:
        processed_dataset_path = directory.interim.joinpath('dataset.csv')

        tune_dataset_path = directory.interim.joinpath('dataset_tune.csv')
        val_dataset_path = directory.interim.joinpath('dataset_validate.csv')
        
        tune_split, val_split = split_dataset(processed_dataset_path, validate_ratio, seed)

        open(tune_dataset_path, 'w').write(tune_split)
        open(val_dataset_path, 'w').write(val_split)


def split_mlflow(config_path: str, run_id: str):
    """
    Split preprocessed dataset, that is stored in MLFlow. All the intermediate results
    are to be stored in MLFlow.

    :param config_path: path to configuration file
    :type config_path: str
    :param run_id: id of run
    :type run_id: str
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
        experiment_name = config['experiment']
        seed = config['dataset']['seed']
        validate_ratio = config['dataset']['validate']

    client = MlflowClient()

    with ContextualizedDirectory() as directory:
        processed_dataset_path = directory.interim.joinpath('dataset.csv')

        tune_dataset_path = directory.interim.joinpath('dataset_tune.csv')
        val_dataset_path = directory.interim.joinpath('dataset_validate.csv')

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id) as run:
            client.download_artifacts(run.info.run_id, 'interim/dataset.csv', directory.data)
            tune_split, val_split = split_dataset(processed_dataset_path, validate_ratio, seed)

            open(tune_dataset_path, 'w').write(tune_split)
            open(val_dataset_path, 'w').write(val_split)

            mlflow.log_artifact(tune_dataset_path, 'interim')
            mlflow.log_artifact(val_dataset_path, 'interim')


@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    split_local(config)
    #split_mlflow(config, MLFLOW_RUN_ID)


if __name__ == '__main__':
    main()
