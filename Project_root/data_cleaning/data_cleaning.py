from data_loading import data_loading
from config import config
import numpy as np
from indicnlp.tokenize import indic_tokenize
from transformers import AutoTokenizer
import os 

def tokenize_src_dst_lang_sentence():
    data = data_loading.load_data()
    tokenizer = AutoTokenizer.from_pretrained(config.SRC_LANG_TOKENIZER_MODEL)
    data[config.COLUMN_NAMES[1]] = data[config.COLUMN_NAMES[1]].apply(tokenizer.tokenize)
    data[config.COLUMN_NAMES[3]] = data[config.COLUMN_NAMES[3]].apply(lambda x: indic_tokenize.trivial_tokenize(x,lang=config.DST_LANG))
    return data

 