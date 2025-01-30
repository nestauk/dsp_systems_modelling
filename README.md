# Complex systems modelling
Building policy evidence base and predicting policy impact

## Installation

We're using [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) for dependency management.

Run the following commands to install depedencies.

```
poetry lock && poetry install --with lint
poetry run pre-commit install
poetry self add poetry-plugin-shell
```

To start the environment in your terminal

```
poetry shell
```

To add a new package, use poetry add:

```
poetry add package-name
```
