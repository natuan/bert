import pandas as pd
import nltk.data
from tqdm import tqdm
from bs4 import BeautifulSoup
import random


def load_job_classification_dataset():
    dataset = pd.read_csv('data/job_classification.csv.gz')
    dataset = dataset[dataset['job_id'].notna()]
    dataset = dataset[dataset['ats_id'].notna()]
    dataset = dataset[dataset['description'].notna()]
    dataset = dataset[dataset['title'].notna()]
    dataset = dataset[dataset['level1_id'].notna()]
    dataset = dataset[dataset['level1_name'].notna()]
    return dataset


def get_job_texts(job_ids,
                  delimiter='¥',
                  data_augmenter=None,
                  text_cleaner=None):
    dataset = load_job_classification_dataset()
    dataset = dataset.set_index('job_id')
    texts = [''] * len(job_ids)
    for idx, job_id in enumerate(tqdm(job_ids, 'Getting job texts')):
        title = dataset.loc[job_id]['title']
        desc = dataset.loc[job_id]['description']
        desc = data_augmenter.transform(desc) if data_augmenter else desc
        if text_cleaner:
            title = text_cleaner.transform([title])[0]
            desc = text_cleaner.transform([desc])[0]
        texts[idx] = title + ' ' + delimiter + ' ' + desc
    return texts


class SentenceSwapper:
    def __init__(self):
        self._sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def transform(self, text):
        """
        Randomly swap sentences inside the input document
        """
        soup = BeautifulSoup(text, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        sents = self._sent_detector.tokenize(text.strip())
        new_sents = random.sample(sents, k=len(sents))
        return ' '.join([s for s in new_sents])
