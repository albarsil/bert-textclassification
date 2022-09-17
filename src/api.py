# This is the file that implements a flask server to do inferences

import logging
import os
import sys

from flask import Flask, abort, jsonify, make_response, request
from pyschemavalidator import validate_param

from predictor import ScoringService

# Create logger
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger()

# The flask app for serving predictions
app = Flask(__name__)

# Create the ScoringService object to make the predictions
prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
model = ScoringService(model_path)
model.init()
 
MODEL_THRESHOLD = 0.5

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare it healthy if we can load the model successfully. This method is mostly used by PicPay stuffs to check if the service is up and running"""
    return make_response("", 200) 

@app.route("/invocations/predict", methods=["POST"])
@validate_param(key="text", keytype=str, isrequired=True)
def invocations():
    """
        Do an inference on a single point of data. In this server, we take data as JSON, parse the input arguments and predict
    """
    
    data = request.get_json(silent=True)

    # Do the prediction
    pred = model.predict(data["text"])
    pred = round(pred,2)

    return make_response(
        jsonify({
            "text": data["text"],
            "score": pred
        }),
        200
    )

# if __name__ == "__main__":
#     app.run(host='0.0.0.0')
