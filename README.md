# BERT TEXT CLASSIFICATION

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
Contains the devops/operations files

### local_test
A directory containing scripts and a configuration to run simple training and inference jobs locally so you can verify that everything is configured correctly.

Possui alguns arquivos:
* __train-local.sh__: instantiate the container configured for training.
* __serve-local.sh__: instantiate the container configured to serve.
* __test-dir__: The directory that is mounted on the container with test data mounted everywhere that matches the schema of the container.

##### Exemplos de chamadas locais:

`./local_test/train_local.sh` to train the algorithm locally

`./local_test/serve_local.sh` to launch a local flask API

### src
Module containing classes and helper functions.
To build a production-grade inference server in the container, we use the following libraries to simplify the service:

1. __nginx__ : https://www.nginx.com/
2. __gunicorn__ : https://gunicorn.org/ 
3. __flask__ https://flask.palletsprojects.com/en/

When SageMaker starts a container, it invokes the container with an argument of __train__ or __serve__. We configure this container so that the argument is treated as the command that the container executes. When training, it will run the included __train__ program, and when __serve__ it will run the __serve__ program.

* __api.py__: The API interface with methods.
* __train__: The main model training program. When building your own algorithm, you will edit it to include your training code.
* __serve__: the wrapper that starts the inference server. In most cases, you can use this file as is.
* __wsgi.py__: The startup shell for individual server workers. This only needs to be changed if you changed where predictor.py is located or named.
* __predictor.py__: This is the file that you modify with the business rules in the algorithm inference.
* __nginx.conf__: The configuration of the nginx server that manages the many workers.

### tests
Contains test files for model functions or code and specific business rule cases. You can run local tests with `pytest -v`.
