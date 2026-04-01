[![Test Python Package](https://github.com/peterbednar/insa/actions/workflows/python-test.yml/badge.svg?event=push)](https://github.com/peterbednar/insa/actions/workflows/python-test.yml)

# Iris Predictor

A simple Python package for predicting iris species.

## Development Setup

Clone the repository and install the package in editable mode with development dependencies:

```
pip install -e ".[dev]"
```

## Running Tests

Run all tests with:

```
pytest
```

## Building the Package

Build the source distribution and wheel:

```
python -m build
```

The generated files will appear in the `dist/` directory.

## Uploading to Test PyPI

Upload the distribution files to Test PyPI:

```
twine upload --repository testpypi dist/*
```

You will be prompted to provide your API token.

## Build Docker image

Build the Docker image:

```
docker build -t uui-iris-predictor .
```

TEST