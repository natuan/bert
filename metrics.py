import os
import pandas as pd
from session import get_session_info

config = get_session_info()

training_session_name = 'drop04_lr03e-5_steps10_ls01'

ckpt_step = 8800

output_dir = os.path.join(config['session_dir'], f'outputs_{training_session_name}')

train_df = pd.read_csv(os.path.join(output_dir, f'train_predict_{ckpt_step}.csv'))
dev_df = pd.read_csv(os.path.join(output_dir, f'dev_predict_{ckpt_step}.csv'))
test_df = pd.read_csv(os.path.join(output_dir, f'test_predict_{ckpt_step}.csv'))

train_acc = len(train_df[train_df['predicted_label'] == train_df['label']]) / len(train_df)
dev_acc = len(dev_df[dev_df['predicted_label'] == dev_df['label']]) / len(dev_df)
test_acc = len(test_df[test_df['predicted_label'] == test_df['label']]) / len(test_df)

print(f'Session output: {output_dir}')
print(f'Train accuracy: {train_acc}')
print(f'Dev accuracy: {dev_acc}')
print(f'Test accuracy: {test_acc}')
