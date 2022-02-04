import os
import pathlib
import numpy as np

TRAIN_ROOT_FOLDER = pathlib.Path().absolute()


def word2vec():
    folder_structure = '{}/data/{}'
    train_folder = folder_structure.format(TRAIN_ROOT_FOLDER, 'train')
    test_folder = folder_structure.format(TRAIN_ROOT_FOLDER, 'test')
    vocabulary = _create_dictionary(train_folder)
    _process_folder(train_folder, vocabulary, 'train_dataset')
    print("Train folder processed")
    _process_folder(test_folder, vocabulary, 'test_dataset')
    print("Test folder processed")


def _create_dictionary(folder, max_examples=10000):
    vocabulary = dict()
    pos_folder = '{}/pos_processed'.format(folder)
    neg_folder = '{}/neg_processed'.format(folder)
    folders_to_analyze = [pos_folder, neg_folder]
    for folder in folders_to_analyze:
        texts_filenames = os.listdir(folder)
        for text_filename in texts_filenames:
            text_filepath = '{}/{}'.format(folder, text_filename)
            with open(text_filepath, 'r') as text_file:
                text = text_file.read()
                for word in text.split(' '):
                    if word in vocabulary:
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
    # Now we have the dictonary, lest process it
    return _process_vocabulary(vocabulary, max_examples)


def _process_vocabulary(raw_vocabulary, max_examples):
    max_examples = min(len(raw_vocabulary), max_examples)
    words_and_counter = list(raw_vocabulary.items())
    words_and_counter.sort(reverse=True, key=lambda word: word[1])
    sorted_words = list(map(lambda x: x[0], words_and_counter))[:max_examples]
    processed_vocabulary = dict(list(zip(sorted_words, range(len(sorted_words)))))
    return processed_vocabulary


def _process_folder(folder_path, vocabulary, filename):
    pos_folder = '{}/pos_processed'.format(folder_path)
    neg_folder = '{}/neg_processed'.format(folder_path)
    pos_vectors = _process_texts(pos_folder, vocabulary)
    pos_labels = np.ones(len(pos_vectors))
    neg_vectors = _process_texts(neg_folder, vocabulary)
    neg_labels = np.zeros(len(neg_vectors))
    vectors = pos_vectors + neg_vectors
    X = np.vstack(vectors)
    labels = np.concatenate((pos_labels, neg_labels))
    filepath_X = '{}/{}_X.npy'.format(folder_path, filename)
    filepath_y = '{}/{}_y.npy'.format(folder_path, filename)
    with open(filepath_X, 'wb') as f:
        np.save(f, X)
    with open(filepath_y, 'wb') as f:
        np.save(f, labels)


def _process_texts(folder_path, vocabulary):
    word_vecs = []
    texts_filepaths = os.listdir(folder_path)
    for text_filename in texts_filepaths:
        text_filepath = '{}/{}'.format(folder_path, text_filename)
        with open(text_filepath, 'r') as text_file:
            text = text_file.read()
            word_vec = _process_text(text, vocabulary)
            word_vecs.append(word_vec)
    return word_vecs


def _process_text(text, vocabulary):
    vector = [0] * len(vocabulary)
    for word in text.split(' '):
        if word in vocabulary:
            vector[vocabulary[word]] += 1
    return np.array(vector)


def _save_processed_vector(word_vec, old_folder_path, text_filename):
    folder_path = '{}_wordvecs'.format(old_folder_path)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    filename = '{}/{}.npy'.format(folder_path, text_filename[:-4])
    with open(filename, 'wb') as f:
        np.save(f, word_vec)


if __name__ == "__main__":
    word2vec()
