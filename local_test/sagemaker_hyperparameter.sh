
PROFILE=$1
IMAGE="bert-textclassification"
SHORTNAME="sfsc"
DT=`date "+%Y%m%d%H%M%S"`
JOBNAME="${SHORTNAME}-${DT}"

echo "Triggering container build and push"
./local_test/build_and_push.sh $IMAGE $PROFILE

echo "Executing job: ${JOBNAME}"
aws sagemaker create-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "${JOBNAME}" \
    --hyper-parameter-tuning-job-config file://ops/config/hyperparameter-job-config.json \
    --training-job-definition file://ops/config/training-job-definition.json \
    --warm-start-config file://ops/config/warm-start-config.json \
    --tags file://ops/tags.json \
    --profile=$PROFILE