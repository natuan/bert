import pandas as pd
import nltk.data
from tqdm import tqdm
from bs4 import BeautifulSoup
import random
import collections


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
                  text_cleaner=None):
    dataset = load_job_classification_dataset()
    dataset = dataset.set_index('job_id')
    texts = [''] * len(job_ids)
    for idx, job_id in enumerate(tqdm(job_ids, 'Getting job texts')):
        title = dataset.loc[job_id]['title']
        desc = dataset.loc[job_id]['description']
        if text_cleaner:
            title = text_cleaner.transform([title])[0]
            desc = text_cleaner.transform([desc])[0]
        texts[idx] = title + ' ' + delimiter + ' ' + desc
    return texts


class TextManipulator_SentSwapper:
    def __init__(self, ratio=1.0):
        self._sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self._ratio = ratio

    def transform(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        sents = self._sent_detector.tokenize(text.strip())
        n = int(self._ratio * len(sents))
        first_sents = sents[:n]
        new_first_sents = random.sample(first_sents, k=len(first_sents))
        new_sents = new_first_sents + sents[n:]
        assert len(new_sents) == len(sents)
        return ' '.join([s for s in new_sents])

    def get_config_info(self):
        return {'name': 'TextManipulator_SentSwapper'}


class DataAugmenter_JobDup:
    def __init__(self,
                 delimiter='¥',
                 text_cleaner=None,
                 text_manipulator=None,
                 max_size_per_class=1000, shrink_oversize_class=False):
        self._delimiter = delimiter
        self._text_cleaner = text_cleaner
        self._text_manipulator = text_manipulator
        self._max_size_per_class = max_size_per_class
        self._shrink_oversize_class = shrink_oversize_class

        self._dataset = load_job_classification_dataset()
        self._dataset = self._dataset.set_index('job_id')


    def _transform_texts(self, job_ids):
        texts = [''] * len(job_ids)
        for idx, job_id in enumerate(tqdm(job_ids, 'Transforming texts')):
            title = self._dataset.loc[job_id]['title']
            desc = self._dataset.loc[job_id]['description']
            if self._text_manipulator:
                desc = self._text_manipulator.transform(desc)
            if self._text_cleaner:
                title = self._text_cleaner.transform([title])[0]
                desc = self._text_cleaner.transform([desc])[0]
            texts[idx] = title + ' ' + self._delimiter + ' ' + desc
        return texts


    def transform(self, df, applied_levels=None):
        """
        Randomly swap sentences inside the input document
        """
        frames = []
        level_group = df.groupby('level1_id')
        for group in level_group:
            level = group[0]
            ids = group[1]['job_id'].tolist()
            applicable = (applied_levels is None) or (level in applied_levels)
            d = collections.defaultdict(list)
            if applicable:
                print(f'Transforming level {level}...')
                if len(ids) >= self._max_size_per_class:
                    d['job_id'] = ids[:self._max_size_per_class] \
                        if self._shrink_oversize_class else ids
                    d['level1_id'] = [level] * len(d['job_id'])
                    d['job_text'] = get_job_texts(d['job_id'],
                                                  self._delimiter,
                                                  self._text_cleaner)
                else:
                    for i in range(int(self._max_size_per_class/len(ids))):
                        d['job_id'] += [f'{j}_{i}' for j in ids]
                        d['level1_id'] += [level] * len(ids)
                        if i == 0:
                            d['job_text'] = get_job_texts(ids,
                                                          self._delimiter,
                                                          self._text_cleaner)
                        else:
                            d['job_text'] += self._transform_texts(ids)
                    rest = self._max_size_per_class % len(ids)
                    if rest > 0:
                        rest_ids = ids[:rest]
                        group_idx = int(self._max_size_per_class/len(ids))
                        d['job_id'] += [f'{j}_{group_idx}' for j in rest_ids]
                        d['level1_id'] += [level] * len(rest_ids)
                        d['job_text'] += self._transform_texts(rest_ids)
            else:
                print(f'Retrieving level {level}...')
                d['job_id'] = ids
                d['level1_id'] = [level] * len(d['job_id'])
                d['job_text'] = get_job_texts(d['job_id'],
                                              self._delimiter,
                                              self._text_cleaner)
            assert len(d['job_id']) == len(d['level1_id']) == len(d['job_text'])
            frames.append(pd.DataFrame(data=d))
        return pd.concat(frames, ignore_index=True)

    def get_config_info(self):
        tm = self._text_manipulator
        return {'name': 'DataAugmenter_JobDup',
                'text_manipulator': tm.get_config_info() if tm else 'None',
                'max_size_per_class': self._max_size_per_class,
                'shrink_oversize_class': self._shrink_oversize_class}
