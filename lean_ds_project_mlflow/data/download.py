import os
import yaml
import click
import mlflow
import pathlib
from loguru import logger

from lean_ds_project_mlflow import ContextualizedDirectory

#UNCOMMENT TO HAVE MLFLOW ACCESS FROM LOCAL MACHINE
#os.environ['MLFLOW_TRACKING_URI'] = 'http://10.80.20.26:5000'

def dowload_dataset(path: str):
    """
    This method implements the logic of dataset downloading from predefined source. In this case, 
    the dataset is assumed to be stored in system directory: path

    :param path: path to dataset
    :type path: str
    """    
    with open(path, 'w') as f:
        f.write('44')


def download_local(config_path: str):
    """
    Download raw data / raw dataset and store it locally in predefined 
    folder (/data/raw) in local filesystem.

    :param config_path: path to configuration file
    :type config_path: str
    """
    with ContextualizedDirectory() as directory:
        fname = directory.raw.joinpath('dataset.csv')
        dowload_dataset(fname)


def download_mlflow(config_path: str, run_id: str=None):
    """
    Download raw data / raw dataset and store it in MLFlow.

    :param config_path: path to configuration file
    :type config_path: str
    :param run_id: id of MLFlow run, defaults to None
    :type run_id: str, optional
    """    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        experiment_name = config['experiment']
    
    with ContextualizedDirectory() as directory:
        fname = directory.raw.joinpath('dataset.csv')
        dowload_dataset(fname)

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id = run_id):
            mlflow.log_artifact(fname, artifact_path='raw')


@logger.catch()
@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config_local.yml')
def main(config):
    download_local(config)
    #download_mlflow(config, None)
                

if __name__ == '__main__':
    main()
