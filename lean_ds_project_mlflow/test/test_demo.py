import os
import yaml
import unittest
from loguru import logger
import mlflow.pyfunc

class TestDemo(unittest.TestCase):
    """
    Test case, all the test methods should be started with test_ prefix!
    """    
    @classmethod
    def setUpClass(cls):
        print('Loading the model...')
        experiment_name = os.getenv("MLFLOW_MODEL_NAME")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE")

        try:
            logger.info('Trying to load model from MLFLow...')
            cls.model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{experiment_name}/{model_stage}"
            )
            logger.info('Model from MLFlow has been loaded.')
        except Exception as e:
            logger.info("Unable to load model from MLFlow due to following reason,")
            logger.exception(e)
            cls.model = mlflow.pyfunc.load_model(model_uri='/models')
            logger.info('Local model has been loaded.')
        finally:
            assert cls.model is not None, "Unable to load model neither from MLflow nor locally."
    
    def test_sanity(self):
        self.assertTrue(True)

    def test_is_model(self):
        self.assertTrue(hasattr(self.model, 'predict'))
        
    def test_expected_behaviour_1(self):
        self.assertEqual(self.model.predict(None), 42)
        
    def test_expected_behaviour_2(self):
        self.assertEqual(self.model.predict('hello'), 42)
    
    def test_expected_behaviour_3(self):
        self.assertEqual(self.model.predict(42), 42)
