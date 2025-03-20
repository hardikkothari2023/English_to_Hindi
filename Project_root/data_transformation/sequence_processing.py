import torch
from config import config

def convert_tokens_to_tensor(data, column_name):
    return torch.tensor(data[column_name].tolist(), dtype=torch.long)

def create_batches(data, batch_size):
    
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
