import logging
import os
import pickle
import sys

import pandas as pd
from transformers import AutoModel, AutoTokenizer

sys.path.append('./src/')
from model import EMBEDDING_MODEL_PATH
from predictor import ScoringService

THRESHOLD = 0.4

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger()

emb_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
emb_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
model_path = './local_test/test_dir/model/'

def test_load_models():
    ss = ScoringService(model_path)

    model = ss._get_model()

    assert model is not None

def test_predict():
    ss = ScoringService(model_path)

    test_cases = [
        'asdaasdasd'
    ]

    out = []
    for x in test_cases:
        out.append(ss.predict(x))

    assert len(out) == len(test_cases)
    assert isinstance(out[0], float)
    assert sum([1 if x is None else 0 for x in out]) == 0
