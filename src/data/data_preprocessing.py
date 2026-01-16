import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import os
import re

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


lemy = WordNetLemmatizer()
stop = stopwords.words('english')
punc = string.punctuation

# Params file input
def load_param(params_path):
    pass

#load the data from folder

def load_data(data_path):
    df = pd.read_csv(data_path)    
    return df

# Basic cleaning
def preprocess_data(train_data, test_data):
    train_data.fillna('', inplace = True)
    test_data.fillna('', inplace = True)
    return train_data, test_data


# transform the data
def lower(text):
    text = str(text)
    if text == 'nan' or text.strip() == '':
        return ''
    
    text = text.lower().strip()
    return text

    # Remove URLs
def remove_url(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    return text


# Tokenize
def tokenize_text(text):
    text = nltk.word_tokenize(text)
    return text

# Keep only alphanumeric tokens
def remove_special_characters(text):
    text = [word for word in text if word.isalnum()]
    return text
# Remove stopwords and punctuation
def remove_stop_puncs(text):
    text = [word for word in text if word not in stop and word not in punc]
    return text

# Lemmatize
def lemmatization(text):
    text = [lemy.lemmatize(word) for word in text]
    return ' '.join(text)

# transform the data

def transform_data(df):

    df['transformed_text'] = df['text'].apply(lambda text: lower(text))
    df['transformed_text'] = df['transformed_text'].apply(lambda text: remove_url(text))
    df['transformed_text'] = df['transformed_text'].apply(lambda text: tokenize_text(text))
    df['transformed_text'] = df['transformed_text'].apply(lambda text: remove_special_characters(text))
    df['transformed_text'] = df['transformed_text'].apply(lambda text: lemmatization(text))
    
    return df
 
def save_data(data_path, train_data, test_data):

    os.makedirs(data_path, exist_ok=True)

    train_data.to_csv(os.path.join(data_path, 'train_processed.csv'))
    test_data.to_csv(os.path.join(data_path, 'test_processed.csv'))

def main():

    train_data = load_data('./data/raw/train.csv')
    test_data = load_data('./data/raw/test.csv')

    train_processed_data = transform_data(train_data)
    test_processed_data = transform_data(test_data)

    data_path = os.path.join('data', 'interim')

    save_data(data_path, train_processed_data, test_processed_data)

if __name__ == '__main__':
    main()