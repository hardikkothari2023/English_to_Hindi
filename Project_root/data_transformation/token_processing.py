from config import config

def add_special_tokens(tokenized_sentence):
     
    return ["<SOS>"] + tokenized_sentence + ["<EOS>"]

def pad_sequences(tokenized_sentences, max_length, pad_token="<PAD>"):
     
    return [sentence[:max_length] + [pad_token] * (max_length - len(sentence)) for sentence in tokenized_sentences]
