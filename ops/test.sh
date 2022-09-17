#!/bin/sh

# install pip requirements
pip --no-cache-dir install -r requirements.txt

# install nltk dependencies
python -m nltk.downloader stopwords punkt

# install git lfs
apt -y update && apt-get -y update && apt-get install -y --no-install-recommends nginx git git-lfs

# create the root dir
mkdir /opt/ml

# copy models to that dir
cp -r local_test/test_dir/model/ /opt/ml/model/

# Huggyface stuffs
mkdir -p /root/.cache/huggingface/transformers/
cd /root/.cache/huggingface/transformers/ && git lfs install && git clone https://huggingface.co/Geotrend/distilbert-base-pt-cased

# copy apy stuffs
cd /drone/src && cp src/nginx.conf /opt/ml/nginx.conf

# Start serving
cd src && python serve &

# Trigger tests
pytest -v