import yaml
import mlflow

def create_run(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.load(f)
        
        experiment_name = config['experiment']
        code_version = config['version']
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param('code', code_version)
        print(code_version)
        run = mlflow.active_run()
        return run.info.run_id
