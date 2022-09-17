#!/bin/sh

image="bert-textclassification"

cp Dockerfile Dockerfiletmp
sed -i '' '$d' Dockerfiletmp

docker build -t ${image} -f Dockerfiletmp .

rm -rf Dockerfiletmp

docker run -it \
    -p 5000:5000 \
    -p 80:80 \
    -v $(pwd)/local_test/test_dir/input/:/opt/ml/input/ \
    -v $(pwd)/local_test/test_dir/output/:/opt/ml/output/ \
    -v $(pwd)/local_test/test_dir/model/:/opt/ml/model/ \
    --rm ${image} bash