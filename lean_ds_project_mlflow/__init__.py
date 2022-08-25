import os
import shutil
import pathlib 
import tempfile

from collections import namedtuple

PYTHONPATH = pathlib.Path.cwd()
IS_AIRFLOW_ENVIRONMENT = bool(os.getenv('AIRFLOW_HOME', None))

class ContextualizedDirectory():
    @staticmethod
    def clear_directory(directory: pathlib.Path):
        for path in directory.glob("**/*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def __init__(self, data_directory='data', models_directory='models', reports_directory='reports'):
        self.data_directory = data_directory
        self.models_directory = models_directory
        self.reports_directory = reports_directory

        if IS_AIRFLOW_ENVIRONMENT:
            self._tempdir_obj = tempfile.TemporaryDirectory()
            self.dir = self._tempdir_obj.name
        else:
            self.dir = PYTHONPATH

    def __enter__(self):
        Directory = namedtuple('Directory', ['data', 'raw', 'interim', 'processed', 'external', 'models', 'reports'])
        directory = Directory(
            pathlib.Path(f'{self.dir}/{self.data_directory}'),
            pathlib.Path(f'{self.dir}/{self.data_directory}/raw'),
            pathlib.Path(f'{self.dir}/{self.data_directory}/interim'),
            pathlib.Path(f'{self.dir}/{self.data_directory}/processed'),
            pathlib.Path(f'{self.dir}/{self.data_directory}/external'),
            pathlib.Path(f'{self.dir}/{self.models_directory}'),
            pathlib.Path(f'{self.dir}/{self.reports_directory}'),
        )
        directory.data.mkdir(parents=True, exist_ok=True)
        directory.raw.mkdir(parents=True, exist_ok=True)
        directory.interim.mkdir(parents=True, exist_ok=True)
        directory.processed.mkdir(parents=True, exist_ok=True)
        directory.external.mkdir(parents=True, exist_ok=True)
        directory.models.mkdir(parents=True, exist_ok=True)
        directory.reports.mkdir(parents=True, exist_ok=True)
        return directory

    def __exit__(self, type, value, traceback):
        if hasattr(self, '_tempdir_obj'):
            self._tempdir_obj.cleanup()
