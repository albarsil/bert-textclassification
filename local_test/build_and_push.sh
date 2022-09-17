#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
PROFILE=$2
TAG=${3:-latest}

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

if [ "$PROFILE" == "" ]
then
    echo "Usage: $0 <profile-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --profile=$PROFILE --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region --profile=$PROFILE)
region=${region:-us-east-1}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${TAG}"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --profile=$PROFILE --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --profile=$PROFILE --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --profile=$PROFILE --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR with the full name.
# Customize Dockerfile to avoid starting with service
cp Dockerfile Dockerfiletmp
sed -i '' '$d' Dockerfiletmp
docker build -t ${image} -f Dockerfiletmp .
rm -rf Dockerfiletmp

# Tag docker container
docker tag ${image} ${fullname}

# Push the image to AWS
docker push ${fullname}