import pandas as pd
from sklearn.model_selection import train_test_split
from config import config
from data_loading import data_loading

def split_dataset(test_size=config.TEST_DATA_FRAC):
    
    data = data_loading.load_data()
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data
