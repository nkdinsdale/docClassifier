import numpy as np
import re
import csv
from pprint import pprint
import matplotlib.pyplot as plt
import string
import word2vec
import gensim
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
import zipfile
import json



# build a sklearn-compatible transformer that is initialised with a word -> vector dictionary.
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter (word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class SumEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter (word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.sum([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter (word2vec.values())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=True)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        '''
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
        '''
        return np.array([
                np.mean([self.word2vec[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

class TfidfEmbeddingVectorizerSum(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter (word2vec.values())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, norm = 'l2')
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.sum([self.word2vec[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

############################# STATS FUNCTIONS ##################################
def make_stats(predicitons, ytest):
    #First calculate the accuracy
    acc = accuracy(predicitons, ytest)
    vals = negs_and_pos(predicitons, ytest)
    recall, precision = recall_and_precision(vals[0], vals[1], vals[2], vals[3])
    return acc, recall, precision

def accuracy(predicitons, ytest):
    score = np.zeros_like(predicitons)
    for l in range(len(predicitons)):
        if predicitons[l] == ytest[l]:
            score[l] = 1
    accuracy = np.sum(score)/len(score)
    return accuracy

def negs_and_pos(predictions, ytest):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for l in range(len(predictions)):
        if predictions[l] == 1 and ytest[l] == 1:
            tp += 1
        if predictions[l] == 1 and ytest[l] == 0:
            fp += 1
        if predictions[l] == 0 and ytest[l] == 1:
            fn += 1
        if predictions[l] == 0 and ytest[l] == 0:
            tn += 1
    return tp, tn, fn, fp

def recall_and_precision(tp, tn, fn, fp):
    recall = tp / ((tp + fn)+0.0000001)
    precision = tp / ((tp + fp)+0.000001)           #to prevent division by zero problems
    return recall, precision


#load in the stop words
with open('stopwords.txt', 'r') as file:
    stopWords =  file.read().splitlines()
#At first we are gonna leave in the stop words because think they might actually give useful information ie We present

#Load in the identifier setences
with open('faceCaptions.txt', 'r') as file:
    identifiers = file.read().splitlines()

with open('noFaceCaptions.txt', 'r') as file:
    nonId = file.read().splitlines()


#Load in the identifier sentences
X, y = [], []
j = 0
identifiers = identifiers[0:5]
for id in identifiers:
    id = id.lower()                                         #Make everything lower case
    id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
    words = id.split()
    if words != []:
        X.append(words)
        y.append('1')
        j += 1
#Load in the none identifier sentences
i = 0
for id in nonId:
    while i < j + 2  :        #So get the same number of each type of sentence
        id = id.lower()                                         #Make everything lower case
        id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
        words = id.split()
        if words != []:
            X.append(words)
            y.append('0')
            i += 1

#Convert to numpy arrays to be able to use fancy idexing etc
X = np.array(X)
y = np.array(y)

#Shuffle X and y and split into testing and training sets
shuffler = np.random.permutation(range(X.shape[0]))
X = X[shuffler]
y = y[shuffler]

# Make some predictions
proportion = int(0.9 * X.shape[0])
Xtest = X[proportion:]
Xtrain = X[0:proportion]
ytest = y[proportion:]
ytrain = y[0:proportion]

# train word2vec on all the texts - both training and test set# train
# we're not using test labels, just texts so this is fine
model = Word2Vec(Xtrain, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

#So far we have a dictionary mapping words --> 100 dimensional vector
#We can use this to build features. The simplest way to do this is by averaging
#word vectors for all words in a text.

#These are trained only on the training data
#Mean vector
# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine

#Then try extra trees and svm
etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_sum = Pipeline([
        ("word2vec vectorizer", SumEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=220))])
etree_w2v_tfidf_sum = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizerSum(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("SVM", SVC())])
svm_w2v_tfdif = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("SVM", SVC())])
svm_w2v_sum = Pipeline([
    ("word2vec vectorizer", SumEmbeddingVectorizer(w2v)),
    ("SVM", SVC())])
svm_w2v_tfdif_sum = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizerSum(w2v)),
    ("SVM", SVC())])

all_models = [
    ('etree_w2v', etree_w2v),
    ('etree_w2v_tfidf', etree_w2v_tfidf),
    ('svm_w2v', svm_w2v),
    ('svm_w2v_tfidf', svm_w2v_tfdif),
    ('etree_w2v_sum', etree_w2v_sum),
    ('etree_w2v_tfidf_sum', etree_w2v_tfidf_sum),
    ('svm_w2v_sum', svm_w2v_sum ),
    ('svm_w2v_tfdif_sum', svm_w2v_tfdif_sum)
]

print ('Benchmarking')
unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))

predictions = etree_w2v_tfidf.fit(Xtrain, ytrain).predict(Xtest)
for i in range(Xtest.shape[0]):
    print (Xtest[i])
    print (predictions[i])
    print (ytest[i])



print ('####################### TEST EXAMPLES #############################')
#Load in definetly unseen examples to test
#Load in the identifier setences
with open('testTrueCap.txt', 'r') as file:
    identifiers = file.read().splitlines()

with open('testFalseCap.txt', 'r') as file:
    nonId = file.read().splitlines()

#Load in the identifier sentences
X, y = [], []
j = 0
for id in identifiers:
    id = id.lower()                                         #Make everything lower case
    id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
    words = id.split()
    X.append(words)
    y.append('1')
    j += 1
#Load in the none identifier sentences
i = 0
for id in nonId:
    id = id.lower()                                         #Make everything lower case
    id = re.sub('['+string.punctuation+']', '', id)         #Strip out the punctuation
    words = id.split()
    X.append(words)
    y.append('0')
    i += 1

X = np.array(X)
y = np.array(y)
#Shuffle X and y and split into testing and training sets
shuffler = np.random.permutation(range(X.shape[0]))
X = X[shuffler]
y = y[shuffler]

predictions = etree_w2v_tfidf.predict(X)
for i in range(X.shape[0]):
    print (X[i])
    print (predictions[i])
    print (y[i])

#Convert to int so can get the accuracy and recall etc
y = np.array(y).astype(int)
predictions = np.array(predictions).astype(int)

acc, recall, precision = make_stats(predictions, y)

print ('Accuracy = ', acc)
print ('Recall = ', recall)
print ('Precision =', precision)
