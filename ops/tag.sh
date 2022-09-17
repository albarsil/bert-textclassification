
# Install git-lfs
apk update && apk add --no-cache git-lfs

# Activate git-lfs
git lfs install

# Update tags
CURRENTBRANCH=$(git rev-parse --abbrev-ref HEAD)
git lfs fetch origin $CURRENTBRANCH
git fetch

echo "Using tag: ${TAG}"

# Check if tag exists
TGEXISTS=$(git tag -l "${TAG}")

if [ -z "${TGEXISTS}" ]; then
    DT=$(date '+%d/%m/%Y')
    git tag -a "${TAG}" -m "build ${BUILD_NUMBER} - ${DT}"
    git push origin $TAG

    DT=$(date '+%d/%m/%Y %H:%M:%S')
    echo "The tag ${TAG} was created at ${DT} with build number: ${BUILD_NUMBER} and commit: ${DRONE_COMMIT}"
    exit 0
else
    echo "Tag ${TAG} already exists. Please edit the .project file to bump the project version"
    exit 1
fi