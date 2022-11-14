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
import time
import pickle
import argparse
import nltk
from numba import njit, prange

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
    newsgroups_test = fetch_20newsgroups(subset="test")
    print(len(newsgroups_test.data), " testing documents loaded.")

    print("Example document:")
    print(newsgroups_train.data[0])

    # Preprocess documents - lemmatization and stemming
    processed_docs = list(map(preprocess, newsgroups_train.data))
    processed_docs_test = list(map(preprocess, newsgroups_test.data))

    print("Example document - lemmatized and stemmed:")
    print(processed_docs[0])

    # Construct dictionary

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    print("Dictionary size: ", len(dictionary))

    # Filter words in documents
    def _filter_docs(processed_docs):
        docs = []
        for doc in processed_docs:
            docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
        return docs

    docs = _filter_docs(processed_docs)
    docs_test = _filter_docs(processed_docs_test)

    print("Example document - filtered:")
    print(docs[0])

    return docs, docs_test, dictionary


# from https://stackoverflow.com/questions/18622781/why-is-numpy-random-choice-so-slow
@njit
def sample_distribution(probs: np.ndarray) -> int:
    x = random.random()
    cumsum = 0
    for i, p in enumerate(probs):
        cumsum += p
        if x < cumsum:
            break
    return i


@njit
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class LDATopicModel:
    def __init__(self, docs, dictionary, topics, alpha, gamma, compute_z_nd=True):
        self.flat_docs = np.concatenate(docs)
        self.docs_lengths = np.array([len(d) for d in docs])
        self.docs_offsets = np.cumsum(self.docs_lengths)
        self.dictionary = dictionary
        self.topics = topics
        self.alpha = alpha
        self.gamma = gamma

        self.doc_cnt = len(docs)
        self.wrd_cnt = len(dictionary)

        if compute_z_nd:
            self.flat_z_nd = self.init_random_topics(self.flat_docs, self.topics)
            self.c_d, self.c_w, self.c = self.compute_counts(
                self.flat_z_nd, self.flat_docs, self.docs_lengths
            )

    @property
    def z_nd(self):
        return np.split(self.flat_z_nd, self.docs_offsets)

    @property
    def docs(self):
        return np.split(self.flat_docs, self.docs_offsets)

    def init_random_topics(self, flat_docs, topics):
        return np.random.randint(0, topics, len(flat_docs))

    @staticmethod
    @njit
    def _compute_counts(flat_z_nd, flat_docs, docs_lengths, topics, wrd_cnt):
        doc_cnt = len(docs_lengths)

        c_d = np.zeros((doc_cnt, topics))
        word_idx = 0
        for d in range(doc_cnt):
            for n in range(docs_lengths[d]):
                k = flat_z_nd[word_idx]
                c_d[d][k] += 1

                # check if k is in range
                assert k >= 0 and k < topics

                word_idx += 1

        c_w = np.zeros((wrd_cnt, topics))
        word_idx = 0
        for d in range(doc_cnt):
            for n in range(docs_lengths[d]):
                w = flat_docs[word_idx]
                k = flat_z_nd[word_idx]
                c_w[w][k] += 1

                word_idx += 1

        c = np.zeros(topics)
        for k in flat_z_nd:
            c[k] += 1

        return c_d, c_w, c

    def compute_counts(self, flat_z_nd, flat_docs, docs_lengths):
        # compute initial counts:
        # - c_d[d][k] ... how many words in document 𝑑 are assigned to topic 𝑘.
        # - c_w[m][k] ... how many times the word 𝑚 is assigned to topic 𝑘 (across all documents).
        # - c[k] ... how many words are assigned to topic 𝑘 (across all documents).

        c_d, c_w, c = LDATopicModel._compute_counts(
            flat_z_nd, flat_docs, docs_lengths, self.topics, self.wrd_cnt
        )
        return c_d, c_w, c

    def save(self, filename):
        # save z_nd using pickle
        with open(filename, "wb") as f:
            pickle.dump(self.flat_z_nd.tolist(), f)

    def load(self, filename):
        with open(filename, "rb") as f:
            flat_z_nd = pickle.load(f)
            self.flat_z_nd = np.array(flat_z_nd)

        # check if the loaded z_nd is valid
        # assert len(self.flat_z_nd) == len(self.flat_docs)
        # assert all([k in range(self.topics) for k in self.flat_z_nd])

        # # recompute counts
        self.c_d, self.c_w, self.c = self.compute_counts(
            self.flat_z_nd, self.flat_docs, self.docs_lengths
        )

    def assign_topics(self, docs, iterations):
        flat_docs = np.concatenate(docs)
        docs_lengths = np.array([len(d) for d in docs])
        docs_offsets = np.cumsum(docs_lengths)

        flat_z_nd = self.init_random_topics(flat_docs, self.topics)
        # z_nd = np.split(flat_z_nd, docs_offsets)

        c_d, _, _ = self.compute_counts(flat_z_nd, flat_docs, docs_lengths)
        for it in range(iterations):
            LDATopicModel._step(
                flat_docs,
                flat_z_nd,
                docs_lengths,
                self.c_w,
                c_d,
                self.c,
                self.wrd_cnt,
                self.topics,
                self.alpha,
                self.gamma,
                True,
            )
        return flat_docs, docs_lengths, c_d
        # LDATopicModel._assign_topics(
        #     flat_docs,
        #     flat_z_nd,
        #     docs_lengths,
        #     self.c_w,
        #     c_d,
        #     self.c,
        #     self.wrd_cnt,
        #     self.topics,
        #     self.alpha,
        #     self.gamma,
        # )

    @staticmethod
    @njit
    def _entropy_topic(c_w, c, gamma, wrd_cnt, k):
        H_k = 0
        for w in range(wrd_cnt):
            p = (gamma + c_w[w][k]) / (wrd_cnt * gamma + c[k])
            H_k -= p * np.log2(p)
        return H_k

    def entropy_topic(self, k):
        return LDATopicModel._entropy_topic(
            self.c_w, self.c, self.gamma, self.wrd_cnt, k
        )

    @staticmethod
    @njit
    def _entropy_data(
        docs: np.ndarray,
        docs_lengths: np.ndarray,
        c_w: np.ndarray,
        c_d: np.ndarray,
        c: np.ndarray,
        wrd_cnt: int,
        topics: int,
        alpha: float,
        gamma: float,
    ) -> float:
        H = 0
        Mgamma = wrd_cnt * gamma
        N_test = len(docs)
        word_probs = np.zeros(N_test)
        word_idx = 0
        for d in range(len(docs_lengths)):
            N_d = docs_lengths[d]
            for n in range(N_d):
                w = docs[word_idx]

                KalphaN_d = topics * alpha + N_d
                for k in range(topics):
                    word_probs[word_idx] += (
                        (alpha + c_d[d][k])
                        / KalphaN_d
                        * (gamma + c_w[w][k])
                        / (Mgamma + c[k])
                    )
                word_idx += 1

        H = -np.sum(np.log2(word_probs)) / N_test

        return H

    def entropy_data(self, flat_docs, docs_lengths, c_d):
        H = LDATopicModel._entropy_data(
            flat_docs,
            docs_lengths,
            self.c_w,
            c_d,
            self.c,
            self.wrd_cnt,
            self.topics,
            self.alpha,
            self.gamma,
        )

        return H

    def top_words(self, k, topn=10):
        # return a list of topn words with the highest probability for topic k
        c_wk = [self.c_w[w][k] for w in range(self.wrd_cnt)]
        words = np.argsort(c_wk)[::-1][:topn]
        return [(self.dictionary[w], c_wk[w]) for w in words]

    @staticmethod
    @njit
    def _step(
        flat_docs: np.ndarray,
        flat_z_nd: np.ndarray,
        docs_lengths: np.ndarray,
        c_w: np.ndarray,
        c_d: np.ndarray,
        c: np.ndarray,
        wrd_cnt: int,
        topics: int,
        alpha: float,
        gamma: float,
        recompute_c_d_only: bool,
    ) -> None:
        word_idx = 0
        # prepare p
        p = np.zeros(topics)
        for d, N_d in enumerate(docs_lengths):
            for n in range(N_d):
                w = flat_docs[word_idx]
                # remove word from counts
                k = flat_z_nd[word_idx]
                c_d[d][k] -= 1
                if not recompute_c_d_only:
                    c_w[w][k] -= 1
                    c[k] -= 1

                # compute probabilities
                for k in range(topics):
                    p[k] = (
                        (alpha + c_d[d][k])
                        * (gamma + c_w[w][k])
                        / (wrd_cnt * gamma + c[k])
                    )
                p = p / np.sum(p)

                # sample new topic from distribution p
                k = sample_distribution(p)
                flat_z_nd[word_idx] = k

                # add word to counts
                c_d[d][k] += 1
                if not recompute_c_d_only:
                    c_w[w][k] += 1
                    c[k] += 1

                word_idx += 1
        assert word_idx == len(flat_docs)

    def step(self):
        LDATopicModel._step(
            self.flat_docs,
            self.flat_z_nd,
            self.docs_lengths,
            self.c_w,
            self.c_d,
            self.c,
            self.wrd_cnt,
            self.topics,
            self.alpha,
            self.gamma,
            False,
        )


def train(
    docs, dictionary, root_dir, iterations, topics, alpha, gamma, seed, save_each=5
):
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # latent dirichlet allocation topic model
    model = LDATopicModel(docs, dictionary, topics, alpha, gamma)

    # create folder for intermediate results
    result_dir = os.path.join(
        root_dir,
        f"ldamodel_topics{topics}_alpha{alpha}_gamma{gamma}_iterations{iterations}_seed{seed}",
    )
    # skip if folder exists
    # if os.path.exists(result_dir):
    #     print(f"Skipping {result_dir}")
    #     return

    os.makedirs(result_dir, exist_ok=True)
    model.save(os.path.join(result_dir, f"model_0.pickle"))

    perplexities = []

    # run iterations
    start_time = time.time()
    for it in range(1, iterations + 1):
        model.step()

        if it % save_each == 0:
            H = model.entropy_data(model.flat_docs, model.docs_lengths, model.c_d)
            PP = 2**H
            perplexities.append((it, PP))
            print("Iteration", it, end=", ")
            print(f"PP={PP}", end=", ")
            print(f"Time={round(time.time() - start_time, 2)}")
            start_time = time.time()
            model.save(os.path.join(result_dir, f"model_{it}.pickle"))

    return perplexities, model


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Resume from pickle checkpoint")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    # set seed for reproducibility
    SEED = 420
    random.seed(SEED)
    np.random.seed(SEED)

    # load cached data if available
    cache_path = "preprocessed_20newsgroups.pickle"
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            docs, docs_test, dictionary = pickle.load(f)
    else:
        docs, docs_test, dictionary = load_data()
        # save docs and dictionary to cache using pickle
        with open(cache_path, "wb") as f:
            pickle.dump((docs, docs_test, dictionary), f)

    # set the hyperparameters
    topics = 20
    alpha = 0.1
    gamma = 0.1

    # latent dirichlet allocation topic model
    model = LDATopicModel(docs, dictionary, topics, alpha, gamma)

    # resume if we have a checkpoint
    if args.resume:
        result_dir = os.path.dirname(args.resume)
        model.load(args.resume)
        start = int(args.resume.split("_")[-1].split(".")[0])
    else:
        # create folder for intermediate results
        result_dir = f"ldamodel_topics{topics}_alpha{alpha}_gamma{gamma}"
        os.makedirs(result_dir, exist_ok=True)
        model.save(os.path.join(result_dir, f"model_0.pickle"))

        start = 0

    # run iterations
    for it in range(start, args.iterations):
        print("Iteration", it + 1)
        start_time = time.time()
        model.step()
        print("Time per iteration:", time.time() - start_time)
        model.save(os.path.join(result_dir, f"model_{it+1}.pickle"))


if __name__ == "__main__":
    main()
