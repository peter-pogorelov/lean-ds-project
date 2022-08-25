import os
import json
import yaml
import click
import typing
import joblib
import mlflow
import mlflow.pyfunc

from loguru import logger
from mlflow.tracking import MlflowClient
from lean_ds_project_mlflow import ContextualizedDirectory


def get_accuracy(X, y, clr) -> float:
    """
    This is a mock method that generates accuracy for classification.

    :param X: observations
    :param y: target variable
    :param clr: classifier to be used for classification
    :return: accuracy metric
    :rtype: float
    """    
    import random
    return random.random()
        

def validate_local(config_path: str):
    """
    Validate the classifier built on the previous step. This method uses validation dataset for evaluating
    the metrics for the model.

    :param config_path: path to config file
    :type config_path: str
    """    
    with ContextualizedDirectory() as directory:
        transformed_dataset_path = directory.processed.joinpath('dataset_validate.bin')
        metrics_path = directory.reports.joinpath('metrics.json')

        dataset = joblib.load(transformed_dataset_path)
        clr = mlflow.pyfunc.load_model(model_uri=str(directory.models))

        metrics = dict()
        metrics[f'accuracy-1'] = get_accuracy(dataset, None, clr)
        metrics[f'accuracy-2'] = get_accuracy(dataset, None, clr)
        
        json.dump(metrics, open(metrics_path, 'w'))


def validate_mlflow(config_path: str, run_id: str) -> typing.Dict[str, float]:
    """
    Validate the classifier built on the previous step and store the data in MLFlow. Data for validation as well as the model
    is taken from MLFlow.

    :param config_path: path to configuration file
    :type config_path: str
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        experiment_name = config['experiment']

    client = MlflowClient()

    with ContextualizedDirectory() as directory:
        transformed_dataset_path = directory.processed.joinpath('dataset_validate.bin')
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id) as run:
            client.download_artifacts(run.info.run_id, 'processed/dataset_validate.bin', directory.data)
            dataset = joblib.load(transformed_dataset_path)
            model = mlflow.pyfunc.load_model(
                model_uri=f"runs:/{run_id}/models"
            )

            metrics = dict()
            
            metrics[f'accuracy-1'] = get_accuracy(dataset, None, model)
            metrics[f'accuracy-2'] = get_accuracy(dataset, None, model)

            for key, val in metrics.items():
                mlflow.log_metric(key, val)

            return metrics


@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    validate_local(config)


if __name__ == '__main__':
    main()
