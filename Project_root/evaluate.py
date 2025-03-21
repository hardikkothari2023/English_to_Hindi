import os
import torch
from torchtext.data.metrics import bleu_score
from config import config
from Models.model import Seq2SeqModel
from transform_pipeline import transform_pipeline
from training.train import create_dataloader

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Path
MODEL_DIR = "Models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "trained_model.pth")

def load_model(src_vocab_size, dst_vocab_size):
     
    model = Seq2SeqModel(src_vocab_size, dst_vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model

def translate_sentence(model, sentence, src_vocab, dst_vocab):
     
    src_tensor = torch.tensor([sentence]).to(device)
    with torch.no_grad():
        output = model(src_tensor, None)  # Inference mode
    output_tokens = output.argmax(dim=-1).squeeze(0).tolist()
    
    # Convert token IDs to words
    idx_to_word = {idx: word for word, idx in dst_vocab.items()}
    translated_sentence = [idx_to_word[token] for token in output_tokens if token in idx_to_word]
    
    return translated_sentence

import torch
from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(model, test_dataloader, src_vocab, dst_vocab):
    
    references = []
    hypotheses = []

    idx_to_word = {idx: word for word, idx in dst_vocab.items()}  # Convert indices to words

    for src_sentence, dst_sentence in test_dataloader:
        src_sentence = src_sentence.to(device)
        dst_sentence = dst_sentence.to(device)

        # Translate source sentence
        translated_tokens = translate_sentence(model, src_sentence[0], src_vocab, dst_vocab)

        # Convert translated tokens to words
        translated_sentence = [idx_to_word[token] for token in translated_tokens if token in idx_to_word]

        # Convert reference tokens to words
        reference_sentence = [[idx_to_word[token] for token in dst_sentence[0].tolist() if token in idx_to_word]]

        # Append for BLEU calculation
        references.append(reference_sentence)
        hypotheses.append(translated_sentence)

    # Calculate BLEU score
    bleu = corpus_bleu(references, hypotheses)
    return bleu


def evaluate():
    
    print("🚀 Evaluating Model...")

    
    _, src_vocab, dst_vocab = transform_pipeline()

  
    print("📥 Loading Trained Model...")
    model = load_model(len(src_vocab), len(dst_vocab))
 
    print("📊 Loading Test Data...")
    _, src_vocab, dst_vocab = transform_pipeline()
    test_dataloader = create_dataloader(_, _, batch_size=1)
 
    print("📈 Calculating BLEU Score...")
    bleu = calculate_bleu(model, test_dataloader, dst_vocab)
    print(f"🌟 BLEU Score: {bleu:.4f}")

if __name__ == "__main__":
    evaluate()
    print("🎉 Evaluation Complete!")