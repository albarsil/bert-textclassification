FROM python:3.7-slim-buster

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
         curl \
         git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install -r requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve codes are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

##### Install NLTK required packages
RUN python -m nltk.downloader stopwords punkt

# Cloning the BERT model to the container
RUN mkdir -p /root/.cache/huggingface/transformers/
RUN git lfs install
RUN git clone https://huggingface.co/Geotrend/distilbert-base-pt-cased /root/.cache/huggingface/transformers/distilbert-base-pt-cased/

# Set up the code in the image
COPY src/ /opt/ml/
COPY local_test/test_dir/model/ /opt/ml/model/
WORKDIR /opt/ml/

RUN chmod u+x /opt/ml/train
EXPOSE 80
EXPOSE 5000

RUN export FLASK_RUN_PORT=80

CMD ["serve"]
