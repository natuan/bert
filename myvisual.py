###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_job_count(df, figsize, sort=False):
    if 'level1_id' not in df.columns:
        raise ValueError('level_id must be a column')
    job_counts = df.groupby('level1_id').job_id.count()
    total_jobs = job_counts.sum()
    order = job_counts.sort_values().index if sort else None
    _, ax = plt.subplots(figsize=figsize)
    ax = sns.countplot('level1_id', data=df, order=order)
    plt.title('Distribution of jobs across level 1 categories')
    ax.set_xlabel('Level 1 category')
    ax.set_ylabel('Number of jobs')
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{}\n({:.1f}%)'.format(int(y), 100*y/total_jobs), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    plt.show()


def _get_accuracy_dataframe(predict_df):
    if 'label' not in predict_df.columns or 'predicted_label' not in predict_df:
        raise ValueError('Invalid columns')
    predict_df['correct'] = predict_df['predicted_label'] == predict_df['label']
    label_group = predict_df.groupby('label')
    correct_counts = label_group.correct.sum()
    job_counts = label_group.job_id.count()
    acc_dict = {}
    acc_dict['label'] = np.unique(predict_df['label'])
    acc_dict['acc'] = [correct_counts[l]/job_counts[l] for l in acc_dict['label']]
    acc_dict['count'] = [job_counts[l] for l in acc_dict['label']]
    accuracy_df = pd.DataFrame.from_dict(data=acc_dict)
    accuracy_df.set_index('label', inplace=True)
    return accuracy_df


def plot_accuracy(predict_df, figsize, title):
    accuracy_df = _get_accuracy_dataframe(predict_df)
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(x=accuracy_df['label'], y=accuracy_df['acc'])
    plt.title(title)
    ax.set_xlabel('Level 1 category')
    ax.set_ylabel('Accuracy')

    for idx, p in enumerate(ax.patches):
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.3f}'.format(accuracy_df.iloc[idx]['acc']), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    plt.show()
    

def plot_train_dev_test_accuracy(train_predict_df,
                                 dev_predict_df,
                                 test_predict_df,
                                 figsize):
    train_acc_df = _get_accuracy_dataframe(train_predict_df)
    dev_acc_df = _get_accuracy_dataframe(dev_predict_df)
    test_acc_df = _get_accuracy_dataframe(test_predict_df)
    fig, axs = plt.subplots(5, 5, figsize=figsize)
    idx = 0
    set_types = ['train', 'dev', 'test']
    colors = ['b', 'g', 'r']

    total_jobs = train_acc_df['count'].sum() + dev_acc_df['count'].sum() + test_acc_df['count'].sum()
    
    for label in train_acc_df.index:
        i = int(idx / 5)
        j = int(idx % 5)
        acc = [train_acc_df.loc[label, 'acc'], dev_acc_df.loc[label, 'acc'], test_acc_df.loc[label, 'acc']]
        for k, s in enumerate(set_types):
            a = acc[k]
            axs[i, j].bar(s, a, color=colors[k])
        rects = axs[i, j].patches
        for k, rect in enumerate(rects):
            height = rect.get_height()
            axs[i, j].text(rect.get_x() + rect.get_width()/2.0, height, '{:.4f}'.format(acc[k]), ha='center', va='bottom')

        axs[i, j].set_ylim(0, 1)
        axs[i, j].set_ybound(0, 1.2)

        job_count = train_acc_df.loc[label, 'count'] + dev_acc_df.loc[label, 'count'] + test_acc_df.loc[label, 'count']
        axs[i, j].title.set_text('Label {}: {} jobs ({:.2f}%)'.format(label, job_count, 100*job_count/total_jobs))
        idx += 1
    for ax in axs.flat:
        ax.set(ylabel='accuracy')
    for ax in axs.flat:
        ax.label_outer()
    fig.subplots_adjust(hspace=.5)
    plt.show()
