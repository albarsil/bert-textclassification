#!/bin/sh

image="bert-textclassification"

docker build -t ${image} -f Dockerfile .

docker run -p 80:80 \
    -v $(pwd)/local_test/test_dir/model/:/opt/ml/model/ \
    --rm ${image} serve