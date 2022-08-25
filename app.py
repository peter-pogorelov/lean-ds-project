import os
import yaml
import mlflow

from lib_msa import MSAAsync, log_handlers
from loguru import logger

model = None

async def setup_environment(self):
    global model

    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

        experiment_name = config['experiment']
        model_stage = config['working_stage']

    try:
        logger.info('Trying to load model from MLFLow...')
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{experiment_name}/{model_stage}"
        )
        logger.info('Model from MLFlow has been loaded.')
    except Exception as e:
        logger.info("Unable to load model from MLFlow due to following reason,")
        logger.exception(e)
        model = mlflow.pyfunc.load_model(model_uri='/models')
        logger.info('Local model has been loaded.')
    finally:
        assert model is not None, "Unable to load model neither from MLflow nor locally."


app = MSAAsync(
    service_name="lean-ds-project-mlflow",
    rabbitmq_url=os.getenv("RABBIT_URL"),
    max_tasks_count=int(os.getenv("MAX_TASKS_COUNT")),
    on_startup = setup_environment
)

@app.callback(logger_arg="logger")
async def predict_model(logger, data):
    return {
        'data': data,
    }

if __name__ == '__main__':
    app.register_log_handler(
        log_handlers.get_elasticsearch_log_handler(
            es_url=os.getenv("ES_URL"),
            name=app.service_name
        )
    )
    app.run()