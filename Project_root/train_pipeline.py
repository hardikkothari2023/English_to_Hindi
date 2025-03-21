import os
import torch
from config import config
from training.train import (
    initialize_model,
    create_dataloader,
    define_optimizer,
    define_loss_function,
    train_one_epoch,
)
from transform_pipeline import transform_pipeline

from Models.model import Seq2SeqModel

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model save path
MODEL_DIR = "Models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "trained_model.pth")

def train_pipeline(num_epochs=10, batch_size=16, learning_rate=0.001):
    

    print("🚀 Starting Training Pipeline...")

    os.makedirs(MODEL_DIR, exist_ok=True)
 
    print("📥 Transforming Data...")
    data, src_vocab, dst_vocab = transform_pipeline()

    src_tensors = torch.tensor([
        sentence + [config.EXTRA_TOKENS_DICT["<PAD>"]] * (config.LONGEST_SRC_SENT_LENGTH - len(sentence)) 
        for sentence in data[config.COLUMN_NAMES[1]]
    ])
    dst_tensors = torch.tensor([
        sentence + [config.EXTRA_TOKENS_DICT["<PAD>"]] * (config.LONGEST_DST_SENT_LENGTH - len(sentence)) 
        for sentence in data[config.COLUMN_NAMES[3]]
    ])
 
    print("📊 Creating Dataloader...")
    train_dataloader = create_dataloader(src_tensors, dst_tensors, batch_size)
 
    print("🧠 Initializing Model...")
    model = initialize_model(len(src_vocab), len(dst_vocab)).to(device)
 
    print("⚙️ Setting Up Training Components...")
    optimizer = define_optimizer(model, learning_rate)
    criterion = define_loss_function()
 
    print("🚀 Training Started...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f"📈 Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}")
 
    print("🔥 Training Completed!")
    print("💾 Saving Trained Model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model Saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    print("🔥 Running Training Pipeline...")
    train_pipeline()
    print("✅ Training Completed!")
