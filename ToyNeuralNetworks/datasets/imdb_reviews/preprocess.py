import os
import pathlib
import re
import string

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import wordnet

TRAIN_ROOT_FOLDER = pathlib.Path().absolute()


def preprocess_dataset():
    folder_structure = '{}/data/{}'
    train_folder = folder_structure.format(TRAIN_ROOT_FOLDER, 'train')
    test_folder = folder_structure.format(TRAIN_ROOT_FOLDER, 'test')
    _process_folder(train_folder)
    _process_folder(test_folder)


def _process_folder(folder_path):
    pos_folder = '{}/pos'.format(folder_path)
    neg_folder = '{}/neg'.format(folder_path)
    _process_texts(pos_folder)
    _process_texts(neg_folder)


def _process_texts(folder_path):
    lemmatizer = nltk.WordNetLemmatizer()
    texts_filepaths = os.listdir(folder_path)
    for text_filename in texts_filepaths:
        text_filepath = '{}/{}'.format(folder_path, text_filename)
        with open(text_filepath, 'r') as text_file:
            text = text_file.read()
            processed_text = _process_text(text, lemmatizer)
            _save_processed_text(processed_text, folder_path, text_filename)


def _process_text(text, lemmatizer):
    lowercase = text.lower()
    stripped_html = BeautifulSoup(lowercase, "html.parser").text
    cleaned_text = re.sub('[%s]' % re.escape(string.punctuation), '', stripped_html)
    text_chunks = cleaned_text.split(' ')
    texts_wo_stopwords_chunks = list(filter(lambda word: word not in stopwords.words(), text_chunks))
    texts_chunks_wo_empty_words = list(filter(lambda word: word, texts_wo_stopwords_chunks))
    lemmatized_chunks = [lemmatizer.lemmatize(word, _get_wordnet_pos(word)) for word in texts_chunks_wo_empty_words]
    lemmatized_sentence = ' '.join(lemmatized_chunks)
    return lemmatized_sentence


def _get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def _save_processed_text(text, old_folder_path, text_filename):
    folder_path = '{}_processed'.format(old_folder_path)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_folder = '{}/{}'.format(folder_path, text_filename)
    with open(file_folder, 'w') as file:
        file.write(text)


if __name__ == "__main__":
    preprocess_dataset()
