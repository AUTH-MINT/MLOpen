from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def ret_self(x):
    return x


def bag_of_words(corpus):
    """
    Create bag of words from corpus
    """
    bow = CountVectorizer(tokenizer=ret_self, preprocessor=ret_self)
    bow.fit(corpus)
    return bow


def fit_tf_idf(corpus):
    """
    create TF/IDF from corpus
    """
    tfidf = TfidfVectorizer(tokenizer=ret_self, preprocessor=ret_self)
    tfidf.fit(corpus)
    return tfidf


def tf_idf(corpus, tfidf):
    """
    Transform a corpus
    """
    tfidf_mtx = tfidf.transform(corpus)
    return tfidf_mtx
