import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from myutils import (get_job_texts,
                     load_job_classification_dataset)
from text_cleaner import TextCleaner


#ROOT_DIR = os.path.join('/', 'Users', 'tnguyen', 'src', 'bert')

ROOT_DIR = os.path.join('/', 'home', 'tnguyen', 'src', 'bert')

SESSION = 'JUL06'
SESSION_DIR = os.path.join(ROOT_DIR, SESSION)

CLEANER_CONFIG = {'punctuation': True,
                  'alphabetic': True,
                  'stop_words': False,
                  'stem_words': False,
                  'state_city': False}

DELIMITER = '¥'

TEST_SIZE = 0.2
DEV_SIZE = 0.2
SEED = 0


def make_session():
    def _make_df(X, y):
        data = {}
        data['job_id'] = X
        data['job_text'] = get_job_texts(X,
                                         delimiter=config['delimiter'],
                                         text_cleaner=TextCleaner(config['cleaner_config']))
        data['level1_id'] = y
        assert len(data['job_id']) == len(data['job_text']) == len(data['level1_id'])
        return pd.DataFrame(data=data)

    config = get_session_info()
    assert not os.path.exists(config['session_dir'])
    print('Making session {}...'.format(config['session']))
    os.makedirs(config['session_dir'])

    X_train, X_dev, X_test, y_train, y_dev, y_test = _create_train_dev_test_job_ids(dev_size=config['dev_size'],
                                                                                    test_size=config['test_size'],
                                                                                    random_state=config['seed'])

    # DEBUGGING = True
    # if DEBUGGING:
    #     n = 3
    #     print('DEBUGGING >>>>>>>>>')
    #     X_train = X_train.iloc[:n]
    #     y_train = y_train.iloc[:n]
    #     X_dev = X_dev.iloc[:n]
    #     y_dev = y_dev.iloc[:n]
    #     X_test = X_test.iloc[:n]
    #     y_test = y_test.iloc[:n]

    #     print('<<<<<<<')
    
    train_df = _make_df(X_train, y_train)
    dev_df = _make_df(X_dev, y_dev)
    test_df = _make_df(X_test, y_test)
    train_df.to_csv(os.path.join(config['session_dir'], 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(config['session_dir'], 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(config['session_dir'], 'test.csv'), index=False)

    session_file_path = os.path.join(config['session_dir'], 'session.json')
    with open(session_file_path, 'w') as f:
        json.dump(config, f)
    print('>> Done')


def get_session_info():
    return {'root_dir': ROOT_DIR,
            'session': SESSION,
            'session_dir': SESSION_DIR,
            'cleaner_config': CLEANER_CONFIG,
            'delimiter': DELIMITER,
            'test_size': TEST_SIZE,
            'dev_size': DEV_SIZE,
            'seed': SEED}


def load_session_datasets():
    config = get_session_info()
    train_df = pd.read_csv(os.path.join(config['session_dir'], 'train.csv'))
    dev_df = pd.read_csv(os.path.join(config['session_dir'], 'dev.csv'))
    test_df = pd.read_csv(os.path.join(config['session_dir'], 'test.csv'))
    return train_df, dev_df, test_df


def _create_train_dev_test_job_ids(dev_size=0.2, test_size=0.2, random_state=0):

    assert test_size > 0.0
    assert 0.0 < dev_size + test_size <= 0.5
    dataset = load_job_classification_dataset()
    X = dataset['job_id']
    y = dataset['level1_id']
    dev_test_size = dev_size + test_size
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y,
                                                                test_size=dev_test_size,
                                                                random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=test_size/dev_test_size,
                                                    random_state=2*random_state)
    assert len(set(X_train.tolist()) & set(X_dev.tolist())) == 0
    assert len(set(X_dev.tolist()) & set(X_test.tolist())) == 0
    assert len(set(X_test.tolist()) & set(X_train.tolist())) == 0
    return X_train, X_dev, X_test, y_train, y_dev, y_test


if __name__ == '__main__':
    make_session()
