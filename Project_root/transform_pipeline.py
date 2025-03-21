import os
import torch
from config import config
from data_cleaning.data_cleaning import tokenize_src_dst_lang_sentence
from data_transformation.data_conversion import (
    convert_src_lang_tokens_to_ids,
    create_dst_language_vocab,
    save_vocabulary
)

# Directory to save transformed data
TRANSFORMED_DATA_DIR = "transformed_data"
os.makedirs(TRANSFORMED_DATA_DIR, exist_ok=True)

def transform_pipeline():
    
    print("🚀 Starting Data Transformation Pipeline...")

    # Step 1: Load & Tokenize Data
    print("📥 Loading and Tokenizing Data...")
    data = tokenize_src_dst_lang_sentence()

    # Step 2: Convert Tokens to IDs
    print("🔢 Converting Tokens to IDs...")
    data = convert_src_lang_tokens_to_ids(data)

    # Step 3: Create Vocabulary for Source & Destination
    print("🔠 Creating Vocabularies...")
    src_vocab = list(set([token for sentence in data[config.COLUMN_NAMES[1]] for token in sentence]))
    dst_vocab = create_dst_language_vocab(data[config.COLUMN_NAMES[3]])

    # Step 4: Save Vocabularies
    print("💾 Saving Vocabulary Files...")
    save_vocabulary(os.path.join(TRANSFORMED_DATA_DIR, config.SRC_LANG_VOCAB_FILENAME), src_vocab)
    save_vocabulary(os.path.join(TRANSFORMED_DATA_DIR, config.DST_LANG_VOCAB_FILENAME), dst_vocab)

    print(f"✅ Transformation Complete! Vocabularies saved in '{TRANSFORMED_DATA_DIR}'.")

    return data, src_vocab, dst_vocab

if __name__ == "__main__":
    print("🔥 Running Transforming Pipeline...")
    transform_pipeline()
