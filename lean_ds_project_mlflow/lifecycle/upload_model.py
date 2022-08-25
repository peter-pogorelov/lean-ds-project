import os
import json
import yaml
import click
import mlflow
import mlflow.pyfunc

from loguru import logger
from lean_ds_project_mlflow import ContextualizedDirectory

def upload_local_model_mlflow(config_path):
    """
    Uploading local model that was built with train.py script to MLFlow

    :param config_path: path to configuration file
    :type config_path: str
    """    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        experiment_name = config['experiment']
        mlflow_uri = config['mlflow_uri']
        version = config['version']

    os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri

    with ContextualizedDirectory() as directory:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_param('version', version)

            with open(directory.reports.joinpath('metrics.json'), 'r') as f:
                metrics = json.load(f)

            for key, val in metrics.items():
                mlflow.log_metric(key, val)

            clr = mlflow.pyfunc.load_model(model_uri=str(directory.models))
            
            mlflow.pyfunc.log_model(
                artifact_path='models',
                python_model=clr._model_impl.python_model, 
                registered_model_name=experiment_name
            )

@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    upload_local_model_mlflow(config)

if __name__ == '__main__':
    main()
