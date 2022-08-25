import os
import yaml
import typing
import requests

from loguru import logger

def submit_metrics(config_path, metrics: typing.Dict[str, float]):
    with open(config_path, 'r') as f:
        config = yaml.load(f)
        
        service_name = config['service']
    
    # this variable should be accessible from airflow instance
    metrics_uri = os.getenv('METRICS_TRACKING_URI')
    
    for metric in metrics.keys():
        result = requests.post(metrics_uri+'/train_validation', params={
            'model': service_name,
            'metric': metric,
            'value': metrics[metric]
        })
        
        logger.info(result.content)
