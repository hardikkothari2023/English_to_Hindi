import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import config
from Models.model import Seq2SeqModel

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(input_dim, output_dim, emb_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
     
    model = Seq2SeqModel(input_dim, output_dim, emb_dim, hidden_dim, num_layers, dropout).to(device)
    return model

def create_dataloader(src_tensors, dst_tensors, batch_size=16):
 
    dataset = TensorDataset(src_tensors, dst_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def define_optimizer(model, learning_rate=0.001):
     
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def define_loss_function():
     
    loss_function = nn.CrossEntropyLoss(ignore_index=config.EXTRA_TOKENS_DICT["<PAD>"])
    return loss_function

def train_one_epoch(model, dataloader, optimizer, criterion):
     
    model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
