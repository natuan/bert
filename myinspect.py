import os
import pandas as pd
import csv
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

csv.register_dialect(
    'etl', delimiter='|', escapechar='\\', doublequote=False, quoting=csv.QUOTE_NONE
)

def create_wrong_prediction_csv(predict_csv, label, output_csv_file_path):
    predict_df = pd.read_csv(predict_csv)
    label_df = predict_df[predict_df['label'] == label]
    wrong_df = label_df[label_df['label'] != label_df['predicted_label']]

    pos_df = pd.read_csv('./positions-2019-07-31-08.csv000.gz', dialect='etl')
    pos_df = pos_df.set_index('job_id')

    level02 = [pos_df.loc[job_id, 'classification_cluster_group'] for job_id in wrong_df['job_id']]
    cat_df = pd.read_csv('./imported_position_categories_sorted.csv', sep='|')
    cat_df = cat_df.set_index('cluster_group')
    wrong_df.insert(len(wrong_df.columns), 'level02',
                    [f'{id}/{cat_df.loc[id, "sub_category_name"]}' for id in level02])
    
    titles = [pos_df.loc[job_id, 'title'] for job_id in wrong_df['job_id']]
    wrong_df.insert(len(wrong_df.columns), 'title', titles)

    descs = [None] * len(wrong_df)
    for idx, job_id in enumerate(wrong_df['job_id'].tolist()):
        desc = pos_df.loc[job_id]['description']
        soup = BeautifulSoup(desc, 'html.parser')
        descs[idx] = ' '.join(soup.stripped_strings)
    wrong_df.insert(len(wrong_df.columns), 'description', descs)
    wrong_df.to_csv(output_csv_file_path, index=False)


def job_title_description_stats():

    def _remove_html(doc, debug=False):
        soup = BeautifulSoup(doc, 'html.parser')
        if debug:
            print(soup.prettify())
        text = ' '.join(soup.stripped_strings)
        return text

    def _word_count(doc, remove_html=True):
        if remove_html:
            doc = _remove_html(doc)
        return len(word_tokenize(doc))

    pos_df = pd.read_csv('./positions-2019-07-31-08.csv000.gz', dialect='etl')
    pos_df = pos_df[pos_df['job_id'].notna()]
    pos_df = pos_df[pos_df['description'].notna()]
    desc_words_counts = pos_df['description'].apply(_word_count, remove_html=True)
    title_words_counts = pos_df['title'].apply(_word_count, remove_html=True)
    word_counts = desc_words_counts + title_words_counts + 1
    print(word_counts.describe())


def main():
    MODEL_DIR = 'JUL17_B/outputs_drop04_lr03e-5_steps10_ls01'
    OUTPUT_DIR = os.path.join(MODEL_DIR, 'wrong_predictions')
    label = 6
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for set_type in ['train', 'dev', 'test']:
        print(f'Generating for {set_type} set...')
        create_wrong_prediction_csv(
            os.path.join(MODEL_DIR, f'{set_type}_predict_8800.csv'),
            label,
            os.path.join(OUTPUT_DIR, f'{set_type}_wrong_predict_8800_label{label}.csv'))
        
    #job_title_description_stats()


if __name__ == "__main__":
    main()
