import os
import yaml
import click
import joblib
import mlflow
import pathlib
import tempfile
import mlflow.pyfunc

from loguru import logger
from mlflow.tracking import MlflowClient
from lean_ds_project_mlflow import ContextualizedDirectory


class CustomPythonModel(mlflow.pyfunc.PythonModel):
    """
    Mock model class that uses PuthonModel interface from MLFlow to make
    it serializable even with extra dependencies like references to another classes out
    of scope of this class.
    """    
    def __init__(self):
        super().__init__()

    def predict(self, context, model_input):
        return 42


def train(dataset):
    """
    Train machine learning model
    :param dataset: dataset for training the model
    :type dataset: any
    :return: model
    :rtype: mlflow.pyfunc.PythonModel
    """    
    model = CustomPythonModel()
    return model


def train_local(config_path: str = None):
    """
    Train a model and register it in MLFlow. Optionally, gridsearch for parameters along with
    validation phase can be moved into this method.

    :param config_path: path to configuration file (unused)
    :type config_path: str
    """

    with ContextualizedDirectory() as directory:
        ContextualizedDirectory.clear_directory(directory.models)
        transformed_dataset_path = directory.processed.joinpath('dataset_tune.bin')
        dataset = joblib.load(transformed_dataset_path)
        
        clr = train(dataset)
        mlflow.pyfunc.save_model(
            path=directory.models,
            python_model=clr,
        )

def train_mlflow(config_path: str, run_id: str):
    """
    Train a model and register it in existing MLFlow run. Suitable for automatic retraning.

    :param config_path: path to configuration file
    :type config_path: str
    :param run_id: id of existing run
    :type run_id: str
    """    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        service_name = config['service']
        experiment_name = config['experiment']

    client = MlflowClient()
    
    with ContextualizedDirectory() as directory:
        transformed_dataset_path = directory.processed.joinpath('dataset_tune.bin')
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id) as run:
            client.download_artifacts(run.info.run_id, 'processed/dataset_tune.bin', directory.data)
            dataset = joblib.load(transformed_dataset_path)
            clr = train(dataset)
            mlflow.log_param("algorithm", "test")
            mlflow.pyfunc.log_model(
                artifact_path='models',
                python_model=clr, 
                registered_model_name=service_name
            )


@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    train_local(config)
                

if __name__ == '__main__':
    main()
