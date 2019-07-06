from collections import defaultdict
import re
import string
from sklearn.base import TransformerMixin
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd

def _remove_html(doc, debug=False):
    soup = BeautifulSoup(doc, 'html.parser')
    if debug:
        print(soup.prettify())
    text = ' '.join(soup.stripped_strings)
    return text


class TextCleaner(TransformerMixin):
    def __init__(self, config=None):
        self._config = config if config else defaultdict(lambda: True)
        usa = pd.read_csv('data/us_cities_states_counties.csv', sep='|')
        city_short_state = set()
        city_full_state = set()
        for city, short_state, full_state in zip(usa['City'],
                                                 usa['State short'],
                                                 usa['State full']):
            city_short_state.add(f'{city} {short_state}'.lower())
            city_full_state.add(f'{city} {full_state}'.lower())
        self._city_state = city_short_state | city_full_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, debug=False):
        """
        X - sequence of documents

        """
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            raise ValueError('Input must be list of documents')
        new_X = []
        for doc in X:
            # get rid of html
            text = _remove_html(doc)

            # split into words
            tokens = word_tokenize(text)

            # convert to lower case
            tokens = [w.lower() for w in tokens]

            if debug:
                print('Lower case:')
                print(tokens)

            if self._config['punctuation']:
                # remove punctuation from each word
                re_punc = re.compile('[%s]' % re.escape(string.punctuation))
                stripped = [re_punc.sub('', w) for w in tokens]
                if debug:
                    print('Removed punctuation:')
                    print(stripped)

            if self._config['alphabetic']:
                # remove remaining tokens that are not alphabetic
                words = [word for word in stripped if word.isalpha()]
                if debug:
                    print('Filtered out non-abc words:')
                    print(words)

            if self._config['stop_words']:
                # filter out stop words
                stop_words = set(stopwords.words('english'))
                words = [w for w in words if w not in stop_words]
                if debug:
                    print('Filtered out stop words:')
                    print(words)

            if self._config['stem_words']:
                # stemming of words
                porter = PorterStemmer()
                words = [porter.stem(word) for word in words]

            # filter out short tokens
            words = [word for word in words if len(word) > 1]

            transformed_doc = ' '.join(words)

            if self._config['state_city']:
                # FIXME: filter out cities and states --- EXPENSIVE
                for city_state in self._city_state:
                    # Note: using regex sub function seems too expensive
                    transformed_doc = transformed_doc.replace(f' {city_state} ', ' ')
                    # This second case to remove city state at the end of the text
                    # It's especially for the title
                    transformed_doc = transformed_doc.replace(f' {city_state}', '')
            
            new_X.append(transformed_doc)
        return new_X
