import io
import os
import yaml
import click
import unittest
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = 5428

tests_failed = False
tests_output = None


class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        global tests_failed
        global tests_output

        if not tests_failed:
            self.send_response(200)
        else:
            self.send_response(500)

        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes(tests_output, "utf-8"))


@click.command()
@click.option('--config', type=click.Path(exists=True), help='path to config file', default='config/config.yml')
def main(config):
    global tests_failed
    global tests_output

    with open(config, 'r') as f:
        config = yaml.safe_load(f)
        service = config['service']

        #if 'mlflow_uri' in config.keys():
        #    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow_uri']

        os.environ["MLFLOW_MODEL_NAME"] = config['experiment']
        os.environ["MLFLOW_MODEL_STAGE"] = config['working_stage']

    import_dir = service.replace('-', '_')

    loader = unittest.TestLoader()
    suite = loader.discover(f'{import_dir}/test')
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            result = unittest.TextTestRunner(verbosity=2, stream=buf).run(suite)
        tests_output = buf.getvalue()
    
    if result.failures or result.errors:
        tests_failed = True
    
    webServer = HTTPServer((HOST, PORT), Server)
    print("Server started http://%s:%s" % (HOST, PORT))
    webServer.serve_forever()

if __name__ == "__main__":
    main()