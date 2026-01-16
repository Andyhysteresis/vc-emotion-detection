import numpy as np
import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer

def load_params(params_path):
    max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']
    return max_features

# fetch the data from data/processed
def load_data(path):
    df = pd.read_csv(path)
    return df

def data_preprocessing(df):
    df.fillna('', inplace = True)
    return df

# Splitting the data 
def data_split(train_data, test_data):

    X_train = train_data['text'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['text'].values
    y_test = test_data['sentiment'].values
    return X_train,y_train,X_test,y_test


def Vectorization(X_train, y_train, X_test, y_test, max_features):

    # Apply Bag of Words
    vectorizer = CountVectorizer(max_features = max_features)

    #Fit the vectorizer onto the training data
    X_train_bow = vectorizer.fit_transform(X_train)

    #Fit the vectorizer onto the test  data
    X_test_bow = vectorizer.transform(X_test)

    # Making the train_df
    train_bow_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
    train_bow_df['label'] = y_train
    # Making the test_df
    test_bow_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
    test_bow_df['label'] = y_test

    return train_bow_df, test_bow_df



# store the data inside data/features

def save_data(data_path, train_df, test_df):

    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, 'train_bow.csv'))
    test_df.to_csv(os.path.join(data_path, 'test_bow.csv'))

def main():
    max_features = load_params('params.yaml')
    train_data = load_data('./data/interim/train_processed.csv')
    test_data = load_data('./data/interim/test_processed.csv')    

    data_path = os.path.join('data', 'processed')

    train_data = train_data.apply(data_preprocessing)
    test_data = test_data.apply(data_preprocessing) 

    X_train, y_train, X_test, y_test = data_split(train_data,test_data)
    
    train_bow, test_bow = Vectorization(X_train,y_train,X_test,y_test, max_features)

    save_data(data_path, train_bow,test_bow)


if __name__ == '__main__':
    main()
