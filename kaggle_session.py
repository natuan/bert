import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from myutils import load_job_classification_dataset


ROOT_DIR = os.path.join('/', 'Users', 'tnguyen', 'src', 'bert')

#ROOT_DIR = os.path.join('/', 'home', 'tnguyen', 'src', 'bert')

SESSION = 'kaggle'
SESSION_DIR = os.path.join(ROOT_DIR, SESSION)

SEED = 0


def make_session():
    def _make_df(X, y):
        return pd.DataFrame(data={'job_id': X, 'level1_id': y})

    config = get_session_info()
    assert not os.path.exists(config['session_dir'])
    print('Making session {}...'.format(config['session']))
    os.makedirs(config['session_dir'])

    session_file_path = os.path.join(config['session_dir'], 'session.json')
    with open(session_file_path, 'w') as f:
        json.dump(config, f)
    print('>> Done')


def get_session_info():
    return {'root_dir': ROOT_DIR,
            'session': SESSION,
            'session_dir': SESSION_DIR,
            'seed': SEED}


def load_session_datasets():
    config = get_session_info()
    train_df = pd.read_csv(os.path.join(config['session_dir'], 'Train_rev1.csv'))
    dev_df = pd.read_csv(os.path.join(config['session_dir'], 'Valid_rev1.csv'))
    test_df = pd.read_csv(os.path.join(config['session_dir'], 'Test_rev1.csv'))
    return train_df, dev_df, test_df


if __name__ == '__main__':
    make_session()
