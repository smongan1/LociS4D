from BruteForceHippo import LociS4DLm
from train import train
import torch
import os

D_MODEL = 384
D_STATE = 1024
D_EMBEDDING = 256
CHARACTERISTIC_MAX_SEQ_LEN = 65536
NUM_LAYERS = 4
#(self, input_size: int, state_dim: int, vocab_size: int, embedding_dim: int, num_substates: int = 8, num_blocks = 4, num_routes: int = 8)
WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(WORKDIR, "model.pth")

def load_model(vocab_size, device):
    # Initialize the model architecture
    model = LociS4DLm(model_dim=D_MODEL,
    state_dim=D_STATE,
    vocab_size=vocab_size, 
    embedding_dim=D_EMBEDDING, 
    num_blocks=NUM_LAYERS, 
    seq_len=CHARACTERISTIC_MAX_SEQ_LEN)
    
    # Load weights with map_location to ensure device compatibility
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Move model to the target device
    model = model.to(device)
    # model = torch.compile(model)
    
    return model

def construct_model(vocab_size, device):
    model =  LociS4DLm(model_dim=D_MODEL,
    state_dim=D_STATE,
    vocab_size=vocab_size, 
    embedding_dim=D_EMBEDDING, 
    num_blocks=NUM_LAYERS, 
    seq_len=CHARACTERISTIC_MAX_SEQ_LEN)
    
    model = model.to(device)
    # model = torch.compile(model)
    return model

def main(use_shakespeare: bool = False):
    if os.path.exists(MODEL_PATH):
        model_constructor = load_model
    else:
        model_constructor = construct_model
    return train(model_constructor, use_shakespeare=use_shakespeare)