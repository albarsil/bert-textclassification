

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import json
import unidecode
from html import unescape


import methods
from model import EMBEDDING_MODEL_PATH, SEED

# Here we define the seed
np.random.seed(SEED)
torch.manual_seed(SEED)

class ScoringService(object):

    def __init__(cls, model_path: str):
        """Receive a path to load the model object for this instance"""
        cls._model_path = model_path

    def init(cls):
        """Initialize the model"""
        if hasattr(cls, '_model') == False:
            cls._model, cls._emb_tokenizer = cls._get_model()

        torch.set_num_threads(1) # Used for tune Pytorch to use multiple worker processes and allow do concurrent model inference throught API instead of model
        return

    def _get_model(cls) -> torch.nn.Module:
        """Get the model object for this instance, loading it if it's not already loaded."""
        
        checkpoint = torch.load(os.path.join(cls._model_path, "model.pth"))

        emb_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
        emb_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)

        model = methods.load_model_state(checkpoint, embedding_model=emb_model)
        model.eval()

        return model, emb_tokenizer

    def predict(cls, input: str) -> float:
        """
        For the input, do the predictions and return them.

        Parameters:
            input (str): The data on which to do the predictions. There will be one prediction per row in the dataframe

        Returns:
            float: The model prediction score for the given sentence
        """
        
        if hasattr(cls, '_model') == False:
            cls._model, cls._emb_tokenizer = cls._get_model()

        # Parse strange characters on input data
        input = unescape(unidecode.unidecode(input).lower())

        return methods.predict(model=cls._model, tokenizer=cls._emb_tokenizer, sentence=input, dynamic_input=True, gpu=False)
