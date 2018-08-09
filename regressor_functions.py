import numpy as np
import re
import string
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from Embeddings import TfidfEmbeddingVectorizer
from sklearn.cross_validation import cross_val_score
from tabulate import tabulate


def idenitifier_regressor(verbose = 1):
    #Build the Regressor, just using the most successful one from the analysis earlier, ExtraTreesClassifier with TfidfVectorizer
    #Load in the identifier setences
    with open('diseaseidentifiers.txt', 'r') as file:
        identifiers = file.read().splitlines()

    with open('nonIdentifiers.txt', 'r') as file:
        nonId = file.read().splitlines()

    #Load in the identifier sentences
    X, y = [], []
    j = 0
    for id in identifiers:
        id = id.lower()                                         #Make everything lower case
        id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
        words = id.split()
        X.append(words)
        y.append(1)
        j += 1
    #Load in the none identifier sentences
    i = 0
    for id in nonId:
        #while i < j + :        #So get the same number of each type of sentence
        id = id.lower()                                         #Make everything lower case
        id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
        words = id.split()
        X.append(words)
        y.append(0)
        i += 1

    #Convert to numpy arrays to be able to use fancy idexing etc
    X = np.array(X)
    y = np.array(y)

    print ('TRAINING THE IDENTIFIER REGRESSOR ')
    flag = 0
    if verbose == 1:
        print ('Benchmarking')
    while flag == 0:        #Repeat until the model fit is good enough
        shuffler = np.random.permutation(range(X.shape[0]))
        X = X[shuffler]
        y = y[shuffler]
        model = Word2Vec(X, size=150, window=5, min_count=2, workers=2)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

        etree_w2v_tfidf = Pipeline([
            ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
            ("extra trees", ExtraTreesRegressor(n_estimators=220))])

        all_models = [
            ('etree_w2v_tfidf', etree_w2v_tfidf)
        ]

        unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
        scores = sorted(unsorted_scores, key=lambda x: -x[1])
        score = scores[0]
        if score[1] > 0.4:
            flag = 1
            if verbose == 1:
                print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))

    fittedModel = etree_w2v_tfidf.fit(X, y)
    return fittedModel


def caption_regressor(verbose = 1):
    #Build the Regressor, just using the most successful one from the analysis earlier, ExtraTreesClassifier with TfidfVectorizer
    #Load in the identifier setences
    with open('faceCaptions.txt', 'r') as file:
        identifiers = file.read().splitlines()

    with open('noFaceCaptions.txt', 'r') as file:
        nonId = file.read().splitlines()

    #Load in the identifier sentences
    X, y = [], []
    j = 0
    for id in identifiers:
        id = id.lower()                                         #Make everything lower case
        id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
        words = id.split()
        X.append(words)
        y.append(1)
        j += 1
    #Load in the none identifier sentences
    i = 0
    for id in nonId:
        #while i < j + :        #So get the same number of each type of sentence
        id = id.lower()                                         #Make everything lower case
        id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
        words = id.split()
        X.append(words)
        y.append(0)
        i += 1

    #Convert to numpy arrays to be able to use fancy idexing etc
    X = np.array(X)
    y = np.array(y)

    print ('TRAINING THE CAPTION REGRESSOR ')
    flag = 0
    if verbose == 1:
        print ('Benchmarking')
    shuffler = np.random.permutation(range(X.shape[0]))
    X = X[shuffler]
    y = y[shuffler]
    model = Word2Vec(X, size=150, window=3, min_count=3, workers=2)
    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

    etree_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesRegressor(n_estimators=220))])

    all_models = [
        ('etree_w2v_tfidf', etree_w2v_tfidf)
    ]

    unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
    scores = sorted(unsorted_scores, key=lambda x: -x[1])
    score = scores[0]
    if verbose == 1:
        print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))

    fittedModel = etree_w2v_tfidf.fit(X, y)
    return fittedModel
