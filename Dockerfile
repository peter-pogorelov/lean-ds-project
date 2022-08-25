FROM python:3.9.1

COPY . /app

WORKDIR /app

ENV RABBIT_URL amqp://guest:guest@localhost
ENV BROKER_URL amqp://guest:guest@localhost/lean-ds-project-mlflow

RUN pip install -r requirements.txt
CMD python app.py
