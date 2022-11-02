#!/usr/bin/env python3

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import random
import os

import nltk

# nltk.download("wordnet")
# nltk.download("omw-1.4")


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def load_data():
    # Load documents
    newsgroups_train = fetch_20newsgroups(subset="train")
    print(len(newsgroups_train.data), " documents loaded.")

    print("Example document:")
    print(newsgroups_train.data[0])

    # Preprocess documents - lemmatization and stemming
    processed_docs = list(map(preprocess, newsgroups_train.data))

    print("Example document - lemmatized and stemmed:")
    print(processed_docs[0])

    # Construct dictionary

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    print("Dictionary size: ", len(dictionary))

    # Filter words in documents

    docs = []
    maxdoclen = 0
    for doc in processed_docs:
        docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
        maxdoclen = max(maxdoclen, len(docs[-1]))

    print("Example document - filtered:")
    print(docs[0])

    print("Maximum document length:", maxdoclen)

    return docs, dictionary


# load cached data if available
cache_path = "data.npy"
if os.path.isfile(cache_path):
    docs, dictionary = np.load(cache_path, allow_pickle=True)
else:
    docs, dictionary = load_data()
    np.save(cache_path, [docs, dictionary])

# Set the hyperparameters

iterations = 100
topics = 20
alpha = 0.01
gamma = 0.01

doc_cnt = len(docs)
wrd_cnt = len(dictionary)
