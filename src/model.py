import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

EMBEDDING_MODEL_PATH = "/root/.cache/huggingface/transformers/distilbert-base-pt-cased/"
SEED = 4578

class TextClassifier(nn.Module):
    """
    A model for Text classification
    """
    
    def __init__(self, n_classes: int, embedding_model: AutoModel, max_length: int, hidden_size: list, dropout_prob: float = 0.2, embeddings_grad: bool = False):
        """
        Initialize the model

        Parameters:
            n_classes (int): The number os model classes
            embedding_model (transformers.AutoModel): The embedding model
            max_length (int): The maximum sentence length that model can handle
            hidden_size (list[int]): The model hidden layers
            dropout_prob (float): The dropout value between the layers
            embeddings_grad (bool): (Default False) If True, transformers.AutoModel.requires_grad=True and then the embedding weights will be updated on backpropagation

        For a detailed reference, please check: torch.nn.Module
        """
        
        super(TextClassifier, self).__init__()
        
        self._drop_prob = dropout_prob
        self._max_length = max_length
        self._n_classes = n_classes
        self._embedding_model = embedding_model
        self._embeddings_grad = embeddings_grad
        self._embedding_model.requires_grad = self._embeddings_grad
        self._hidden_layers = hidden_size
        self._hidden_size = [self._embedding_model.config.hidden_size] + hidden_size

        if len(self._hidden_size) > 1:
            layers = [
                nn.Sequential(
                    nn.Linear(in_units, out_units),
                    nn.ReLU()
                )
                for in_units, out_units in zip(self._hidden_size, self._hidden_size[1:])
            ]

            self.fc = nn.Sequential(*layers)       
        else:
            self.fc = None

        if dropout_prob is None:
          self.fc_out = nn.Sequential(
              nn.Linear(self._hidden_size[-1], n_classes)
          )
          
        else:
          self.fc_out = nn.Sequential(
              nn.Dropout(self._drop_prob),
              nn.Linear(self._hidden_size[-1], n_classes)
          )
        
    def forward(self, input_ids, attention_mask):
        out = self._embedding_model(input_ids=input_ids,attention_mask=attention_mask)

        hidden_state = out[0]
        out = hidden_state[:, 0]        

        if self.fc is not None:
            out = self.fc(out)

        return torch.sigmoid(self.fc_out(out))

    @property
    def max_length(self) -> int:
        """
        Get's the maximum sentence length that model can handle

        Returns:
            int: The maximum sentence length that model can handle
        """
        return self._max_length

    def arch(self) -> dict:
        """
        Get's the model architecture

        Returns:
            dict: The model architecture
        """

        return {
            "n_classes": self._n_classes,
            "dropout_prob": self._drop_prob,
            "max_length": self._max_length,
            "embeddings_grad": self._embeddings_grad,
            "hidden_size": ';'.join([str(x) for x in self._hidden_layers])
        }
