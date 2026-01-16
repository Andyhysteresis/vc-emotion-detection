import numpy as np
import pandas as pd
import os

import yaml

import warnings as w
w.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

def load_params(params_path: str) -> float:
    test_size=yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size']
    return test_size


def read_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding ='latin1')
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['textID', 'selected_text', 'Time of Tweet',
       'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)',
       'Density (P/Km²)']

    df.drop(columns = cols, inplace = True)

    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'neutral': 0, 'negative': -1})

    return df

def save_data(data_path, train_data, test_data):

    os.makedirs(data_path, exist_ok=True)

    train_data.to_csv(os.path.join(data_path, 'train.csv'))

    test_data.to_csv(os.path.join(data_path, 'test.csv'))


def main():
    test_size = load_params('params.yaml')
    df = read_data(r'D:/MLOps/SentimentAnalysis/sentiment_analysis_dataset.csv')
    final_df = process_data(df)

    train_data, test_data = train_test_split(final_df, test_size = test_size, random_state = 42)

    data_path = os.path.join('data', 'raw')

    save_data(data_path, train_data, test_data)


if __name__ == '__main__':
    main()



