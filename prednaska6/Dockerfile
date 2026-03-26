FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY README.md ./
COPY src ./src

RUN pip install --upgrade pip
RUN pip install .

# RUN python -m uui_iris_predictor.train_pipeline

ENTRYPOINT ["python", "-m", "uui_iris_predictor.endpoint"]
