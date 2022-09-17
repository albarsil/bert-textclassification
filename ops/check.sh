
# Update tags
git fetch

echo "Checking tag: ${TAG}, build number: ${BUILD_NUMBER} and commit: ${DRONE_COMMIT}"

# Check if tag exists
TGEXISTS=$(git tag -l "${TAG}")

if [ -z "${TGEXISTS}" ]; then
    echo "Tag ${TAG} not exists."
    exit 0
else
    echo "Tag ${TAG} already exists. Please edit the .project file to bump the project version"
    exit 1
fi