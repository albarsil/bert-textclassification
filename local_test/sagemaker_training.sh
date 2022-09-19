
SHORTNAME=$1
PROFILE=$2
IMAGE=$3

DT=`date "+%Y%m%d%H%M%S"`
JOBNAME="${SHORTNAME}-${DT}"

echo "Triggering container build and push"
./local_test/build_and_push.sh $IMAGE $SHORTNAME $PROFILE

echo "Executing job: ${JOBNAME}"
aws sagemaker create-training-job \
    --training-job-name "${JOBNAME}" \
    --hyper-parameters file://local_test/test_dir/input/config/hyperparameters.json \
    --algorithm-specification file://ops/config/training-job-definition.json \
    --tags file://ops/tags.json \
    --profile=$PROFILE