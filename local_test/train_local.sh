#!/bin/sh

image="bert-textclassification"

cp Dockerfile Dockerfiletmp
sed -i '' '$d' Dockerfiletmp

docker build -t ${image} -f Dockerfiletmp .

rm -rf Dockerfiletmp

docker run -it \
    -v $(pwd)/local_test/test_dir/input/:/opt/ml/input/ \
    -v $(pwd)/local_test/test_dir/output/:/opt/ml/output/ \
    -v $(pwd)/local_test/test_dir/model/:/opt/ml/model/ \
    -v $(pwd)/local_test/test_dir/checkpoints/:/opt/ml/checkpoints/ \
    ${image} train