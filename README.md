# BERT TEXT CLASSIFICATION

## Introduction

This repository seeks to be an example of how to create a BERT-like model for text classification and deploy it on a container that can be used by (API Gateway + Lambda + SageMaker endpoint) or other cloud vendors (e.g., Azure, GCP). Currently, this repository seeks to be applied for binary text classification.

We used the tips from Roblox post [How We Scaled Bert To Serve 1+ Billion Daily Requests on CPUs](https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26) to run the inference on CPU units, as its cheaper than using GPU for serving.

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
Contains devops/operation files

### local_test
A directory containing scripts and configurations to trigger training and inference jobs locally.

* __train-local.sh__: trigger the local training container.
* __serve-local.sh__: trigger the local serving container and launch a local flask API.
* __test-dir__: The directory that is mounted on the container with test data mounted everywhere that matches the schema of the container.
* __build_and_push.sh__: A script to trigger the container build and then push it to the AWS SageMaker.
* __sagemaker_training.sh__: Triggers the SageMaker training job with the parameters defined.
* __sagemaker_hyperparameter.sh__: Triggers a SageMaker Hyperparamter Training Job to optimize the algorithm parameters

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
