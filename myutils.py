import pandas as pd


def load_job_classification_dataset():
    dataset = pd.read_csv('data/job_classification.csv.gz')
    dataset = dataset[dataset['job_id'].notna()]
    dataset = dataset[dataset['ats_id'].notna()]
    dataset = dataset[dataset['description'].notna()]
    dataset = dataset[dataset['title'].notna()]
    dataset = dataset[dataset['level1_id'].notna()]
    dataset = dataset[dataset['level1_name'].notna()]
    return dataset


def get_job_texts(job_ids, delimiter, text_cleaner=None):
    dataset = load_job_classification_dataset()
    dataset = dataset.set_index('job_id')
    texts = [''] * len(job_ids)
    for idx, job_id in enumerate(job_ids):
        title = dataset.loc[job_id]['title']
        desc = dataset.loc[job_id]['description']
        if text_cleaner:
            title = text_cleaner.transform([title])[0]
            desc = text_cleaner.transform([desc])[0]
        texts[idx] = title + ' ' + delimiter + ' ' + desc
    return texts
