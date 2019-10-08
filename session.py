import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import myutils
from text_cleaner import TextCleaner


#ROOT_DIR = os.path.join('/', 'Users', 'tnguyen', 'src', 'bert')

ROOT_DIR = os.path.join('/', 'home', 'tnguyen', 'src', 'bert')

SESSION = 'OCT_04'
SESSION_DIR = os.path.join(ROOT_DIR, SESSION)

CLEANER_CONFIG = {'punctuation': False,
                  'alphabetic': False,
                  'stop_words': False,
                  'stem_words': False,
                  'state_city': False}

DELIMITER = '¥'
TEST_SIZE = 0.2
DEV_SIZE = 0.2
STRATIFIED = True
SEED = 12345

def make_session():
    def _make_df(X, y, debug=False):
        data = {}
        data['job_id'] = X
        data['job_text'] = myutils.get_job_texts(X,
                                                 delimiter=config['delimiter'],
                                                 text_cleaner=TextCleaner(config['cleaner_config']))
        data['level1_id'] = y
        assert len(data['job_id']) == len(data['job_text']) == len(data['level1_id'])
        return pd.DataFrame(data=data)

    def _make_augmented_df(X, y):
        data_augmenter = DataAugmenter_JobDup(
            delimiter='¥',
            text_cleaner=TextCleaner(config['cleaner_config']),
            text_manipulator=TextManipulator_SentSwapper(ratio=0.5),
            max_size_per_class=2000,
            shrink_oversize_class=False)
        org = pd.DataFrame(data = {'job_id': X, 'level1_id': y})
        return data_augmenter.transform(org, applied_levels=[20])

    config = get_session_info()
    assert not os.path.exists(config['session_dir'])
    print('Making session {}...'.format(config['session']))
    os.makedirs(config['session_dir'])

    X_train, X_dev, X_test, y_train, y_dev, y_test = _create_train_dev_test_job_ids(dev_size=config['dev_size'],
                                                                                    test_size=config['test_size'],
                                                                                    random_state=config['seed'])

    train_df = _make_df(X_train, y_train) #_make_augmented_df(X_train, y_train)
    dev_df = _make_df(X_dev, y_dev)
    test_df = _make_df(X_test, y_test)
    train_df.to_csv(os.path.join(config['session_dir'], 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(config['session_dir'], 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(config['session_dir'], 'test.csv'), index=False)

    session_file_path = os.path.join(config['session_dir'], 'session.json')
    with open(session_file_path, 'w') as f:
        json.dump(config, f)
    print('>> Done')


def load_session_datasets():
    config = get_session_info()
    train_df = pd.read_csv(os.path.join(config['session_dir'], 'train.csv'))
    dev_df = pd.read_csv(os.path.join(config['session_dir'], 'dev.csv'))
    test_df = pd.read_csv(os.path.join(config['session_dir'], 'test.csv'))
    return train_df, dev_df, test_df


def get_session_info():
    return {'root_dir': ROOT_DIR,
            'session': SESSION,
            'session_dir': SESSION_DIR,
            'cleaner_config': CLEANER_CONFIG,
            'delimiter': DELIMITER,
            'test_size': TEST_SIZE,
            'dev_size': DEV_SIZE,
            'stratified': STRATIFIED,
            'seed': SEED}


def _create_train_dev_test_job_ids(dev_size=0.2, test_size=0.2, random_state=0):
    assert test_size > 0.0
    assert 0.0 < dev_size + test_size <= 0.5

    config = get_session_info()
    dataset = myutils.load_job_classification_dataset()
    X = dataset['job_id']
    y = dataset['level1_id']
    dev_test_size = dev_size + test_size

    strat_y = y if config['stratified'] else None
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, stratify=strat_y,
                                                                test_size=dev_test_size,
                                                                random_state=random_state)

    strat_y_dev_test = y_dev_test if config['stratified'] else None
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test,
                                                    stratify=strat_y_dev_test,
                                                    test_size=test_size/dev_test_size,
                                                    random_state=2*random_state)
    assert len(set(X_train.tolist()) & set(X_dev.tolist())) == 0
    assert len(set(X_dev.tolist()) & set(X_test.tolist())) == 0
    assert len(set(X_test.tolist()) & set(X_train.tolist())) == 0
    return X_train, X_dev, X_test, y_train, y_dev, y_test


if __name__ == '__main__':
    make_session()
