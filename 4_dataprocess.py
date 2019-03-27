import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
import numpy as np


max_features = 20000
maxlen = 100

train = pd.read_csv("./cache/train-processed.csv")
test = pd.read_csv("./cache/predict-processed.csv")

list_sentences_train = train.Discuss.values
list_sentences_test = test.Discuss.values

y = train.Score.values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

EMBEDDING_FILE = './cache/char_vector_128.txt'


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf-8'))

# 一个小bug
print(len(embeddings_index))
del embeddings_index[str(len(embeddings_index) - 1)]

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
print(emb_mean, emb_std)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 128))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


x_train, x_val, y_train, y_val = train_test_split(X_t, y, test_size=0.1, random_state=42)

# 保存中间文件，模型调用
np.savez("./cache/data.npz", x_train, y_train, x_val, y_val, X_te, embedding_matrix)
