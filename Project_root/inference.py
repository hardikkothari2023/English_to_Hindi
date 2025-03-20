import os
import torch
from config import config
from Models.model import Seq2SeqModel
from transform_pipeline import transform_pipeline

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Path
MODEL_DIR = "Models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "trained_model.pth")

def load_model(src_vocab_size, dst_vocab_size):
    """
    Loads the trained Seq2Seq model for inference.

    Parameters:
        src_vocab_size (int): Vocabulary size of the source language.
        dst_vocab_size (int): Vocabulary size of the target language.

    Returns:
        model (Seq2SeqModel): Loaded model in evaluation mode.
    """
    model = Seq2SeqModel(src_vocab_size, dst_vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model

def preprocess_input(sentence, src_vocab):
    """
    Tokenizes and converts the input sentence into numerical IDs.

    Parameters:
        sentence (str): Input sentence in source language.
        src_vocab (dict): Source vocabulary mapping.

    Returns:
        tensor (torch.Tensor): Tokenized and numericalized input sentence.
    """
    # Tokenize the sentence
    tokens = sentence.lower().split()  # Basic tokenization (adjust if necessary)
    
    # Convert words to corresponding IDs using the source vocabulary
    sentence_ids = [src_vocab.get(token, config.EXTRA_TOKENS_DICT["<UNK>"]) for token in tokens]
    
    # Add padding if necessary
    while len(sentence_ids) < config.LONGEST_SRC_SENT_LENGTH:
        sentence_ids.append(config.EXTRA_TOKENS_DICT["<PAD>"])
    
    return torch.tensor([sentence_ids]).to(device)

def translate_sentence(model, sentence, src_vocab, dst_vocab):
    """
    Translates an input sentence from source to target language.

    Parameters:
        model (Seq2SeqModel): The trained translation model.
        sentence (str): Input sentence in source language.
        src_vocab (dict): Source vocabulary.
        dst_vocab (dict): Target vocabulary.

    Returns:
        translated_sentence (str): Translated sentence in the target language.
    """
    # Preprocess the input sentence
    src_tensor = preprocess_input(sentence, src_vocab)

    with torch.no_grad():
        output = model(src_tensor, None)  # Perform inference

    output_tokens = output.argmax(dim=-1).squeeze(0).tolist()
    
    # Convert token IDs to words
    idx_to_word = {idx: word for word, idx in dst_vocab.items()}
    translated_tokens = [idx_to_word[token] for token in output_tokens if token in idx_to_word]

    return " ".join(translated_tokens)

def run_inference():
    """
    Runs the inference pipeline:
    1. Loads the trained model.
    2. Takes user input.
    3. Translates and displays output.
    """
    print("🚀 Running Inference...")

    # Step 1: Load vocabulary
    _, src_vocab, dst_vocab = transform_pipeline()

    # Step 2: Load trained model
    print("📥 Loading Trained Model...")
    model = load_model(len(src_vocab), len(dst_vocab))

    # Step 3: User Input Loop
    while True:
        user_input = input("\n📝 Enter a sentence in English (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("👋 Exiting...")
            break
        
        # Step 4: Translate
        translated_sentence = translate_sentence(model, user_input, src_vocab, dst_vocab)
        
        # Step 5: Display Output
        print(f"💬 Translated Sentence: {translated_sentence}")

if __name__ == "__main__":
    run_inference()
