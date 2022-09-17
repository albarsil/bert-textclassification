import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from model import TextClassifier


def save_model_state(model:torch.nn.Module) -> dict:
    """
    Get a dictionary with the model architecture and hidden state

    Parameters:
        model (TextClassifier): The model

    Returns:
        dict: A dictionary with the model structure and hidden state
    """

    arch = model.arch()
    arch["state_dict"] = model.state_dict()
    return arch

def load_model_state(checkpoint:dict, embedding_model:AutoModel) -> TextClassifier:
    """
    Load the model

    Parameters:
        checkpoint (dict): A torch checkpoint with the model architecture and hidden state
        embedding_model (AutoModel): The AutoModel used by the current TextClassifier model

    Returns:
        TextClassifier: The inference ready model
    """

    state_dict = checkpoint["state_dict"]
    del checkpoint["state_dict"]

    checkpoint["hidden_size"] = [int(x) for x in checkpoint["hidden_size"].split(';')]
    model = TextClassifier(**checkpoint, embedding_model=embedding_model)

    model.load_state_dict(state_dict)
    model.eval()

    return model

def predict(model:torch.nn.Module, tokenizer:AutoTokenizer, sentence:str, dynamic_input:bool=False, gpu:bool=False) -> float:
    """
    Do the model inference

    Parameters:
        model (TextClassifier): The model
        tokenizer (AutoTokenizer): The AutoTokenizer instance used by the current TextClassifier model
        sentence (str): The sentence which will be classified by the model
        dynamic_input (bool): If True, enable the use of dynamic inputs and will not pad the sentence until model.max_length
        gpu (bool): (default False) If True, use torch gpu tensors (Note: your model also should be on gpu)

    Returns:
        float: The model prediction score for the input sentence
    """
    sentence = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=model.max_length,
        return_token_type_ids=True,
        padding=False if dynamic_input == True else 'max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    if gpu:
        return model(input_ids=sentence['input_ids'].cuda(), attention_mask=sentence['attention_mask'].cuda())
    else:
        return model(input_ids=sentence['input_ids'].to(torch.device("cpu")), attention_mask=sentence['attention_mask'].to(torch.device("cpu"))).tolist()[0][0]

def create_data_loader(inputs:np.array, targets:np.array, tokenizer:AutoTokenizer, max_len:int, batch_size:int) -> torch.utils.data.DataLoader:
    """
    Creates a dataset loader

    Parameters:
        inputs (np.array): The inputs
        targets (np.array): The targets
        tokenizer (AutoTokenizer): The AutoTokenizer instance used by the current TextClassifier model
        max_len (int): The maximum sentence length that model can handle
        batch_size (int): The batch size

    Returns:
        torch.utils.data.DataLoader: The batched dataset loader instance
    """

    ds = PreProccesDataset(sentence=inputs, targets=targets, tokenizer=tokenizer, max_len=max_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, drop_last=True)

class PreProccesDataset(torch.utils.data.Dataset):
    """
    A class that is used by torch.utils.data.DataLoader to preprocess each sentence
    For a detailed reference, please check: torch.utils.data.Dataset
    """
    
    def __init__(self, sentence, targets, tokenizer, max_len):
        self.sentence = sentence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, item):
        inputs = [str(self.sentence[item])]
        target = self.targets[item]
        
        # Encodamos todas as entradas
        encoding = [
                    self.tokenizer.encode_plus(
                        x,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        return_token_type_ids=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_tensors='pt',
                        truncation=True
                    )
                  for x in inputs
        ]

        # Contatenamos os ids e attention_masks das sentencas e entidades\n",
        input_ids = torch.cat(tuple(x['input_ids'].flatten() for x in encoding), dim=0)
        attention_mask = torch.cat(tuple(x['attention_mask'].flatten() for x in encoding), dim=0)

        return {
          'input_text': inputs,
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'targets': torch.tensor(target, dtype=torch.long)
        }
