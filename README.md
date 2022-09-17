# BERT TEXT CLASSIFICATION

## Introduction

This repository seeks to be an example about how to create a BERT like model for text classification and deploy it on a common AWS cloud stack (API Gateway + Lambda + SageMaker endpoint).

## Repository structure

```
bert-textclassification/
├─ ops/
│  ├─ config/
│  ├─ *.py (to be run locally)
├─ local_test/
│  ├─ test_dir/
│  │  ├─ input/
│  │  │  ├─ config/
│  │  │  ├─ data/
│  │  │  │  ├─ train/
│  │  │  │  ├─ test/
│  │  │  │  ├─ validation/
│  │  ├─ model/
│  │  ├─ output/
│  ├─ *.sh (to be run locally or trigger cloud executions)
├─ src/
│  ├─ *.py
├─ tests/
│  ├─ *.py (to be run locally or trigger by drone/git)
├─ README.md
```

### ops
Contains the devops/operation files

### local_test
A directory containing scripts and configurations to trigger training and inference jobs locally.

Possui alguns arquivos:
* __train-local.sh__: trigger the container configured for training.
* __serve-local.sh__: trigger the container configured to serve.
* __test-dir__: The directory that is mounted on the container with test data mounted everywhere that matches the schema of the container.

##### Exemplos de chamadas locais:

`./local_test/train_local.sh` to train the algorithm locally

`./local_test/serve_local.sh` to launch a local flask API

### src
Module containing classes and helper functions.
We use the following libraries to create a production ready inference server container:

1. __nginx__ : https://www.nginx.com/
2. __gunicorn__ : https://gunicorn.org/ 
3. __flask__ https://flask.palletsprojects.com/en/

When SageMaker starts a container, it invokes the container with an argument of __train__ or __serve__. We configure this container to receive the operation as an argument that will be executed. The scripts on source folders are the following:

* __api.py__: The API interface with methods.
* __train__: The main model training script. When building your own algorithm, you will edit it to include your training code.
* __serve__: the wrapper that starts the inference server. In most cases, you can use this file as is.
* __wsgi.py__: The startup shell for individual server workers. This only needs to be changed if you changed where predictor.py is located or if it was renamed.
* __predictor.py__: This is the file where you can include your business rules before or after the model inference.
* __nginx.conf__: The configuration of the nginx server.

### tests
Contains test files for model functions or code and specific business rule cases. You can run local tests with `pytest -v`.
