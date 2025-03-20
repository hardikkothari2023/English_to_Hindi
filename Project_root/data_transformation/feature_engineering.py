from config import config
from data_cleaning import data_cleaning
from data_transformation import data_conversion, token_processing
from data_transformation.sequence_processing import convert_tokens_to_tensor
from data_transformation.dataset_preparation import split_dataset

def process_data():
    # Step 1: Load & tokenize
    data = data_cleaning.tokenize_src_dst_lang_sentence()
    
    # Step 2: Convert tokens to IDs
    data = data_conversion.convert_src_lang_tokens_to_ids(data)
    
    # Step 3: Apply special tokens
    data[config.COLUMN_NAMES[1]] = data[config.COLUMN_NAMES[1]].apply(token_processing.add_special_tokens)
    data[config.COLUMN_NAMES[3]] = data[config.COLUMN_NAMES[3]].apply(token_processing.add_special_tokens)
    
    # Step 4: Pad sequences
    data[config.COLUMN_NAMES[1]] = token_processing.pad_sequences(data[config.COLUMN_NAMES[1]], config.LONGEST_SRC_SENT_LENGTH)
    data[config.COLUMN_NAMES[3]] = token_processing.pad_sequences(data[config.COLUMN_NAMES[3]], config.LONGEST_DST_SENT_LENGTH)
    
    # Step 5: Create & save vocabulary
    Vd = data_conversion.create_dst_language_vocab(data[config.COLUMN_NAMES[3]])
    data_conversion.save_vocabulary(config.DST_LANG_VOCAB_FILENAME, Vd)
    
    # Step 6: Convert tokens to PyTorch tensors
    data[config.COLUMN_NAMES[1]] = convert_tokens_to_tensor(data, config.COLUMN_NAMES[1])
    data[config.COLUMN_NAMES[3]] = convert_tokens_to_tensor(data, config.COLUMN_NAMES[3])
    
    return data, Vd
