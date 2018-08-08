import re
import numpy as np
import csv
from pprint import pprint
import matplotlib.pyplot as plt
import string
import word2vec
import gensim
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
import argparse
import os

#Project files
from Embeddings import TfidfEmbeddingVectorizer
from re_functions import *

################################################################################
# Get the file name from the input argument
# Input parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', required=True, help="path to g file")
args = parser.parse_args()
file_pth = args.file
assert os.path.isfile(file_pth)

#Read in the pdf structure in the form of a json --> dictionary of dictionaries
print ('Reading in file:')
with open(file_pth) as json_data:
    e = json.load(json_data)
    json_data.close()

#Search the sections to try and find the sentence, its only going to be on page one or two
sectionDict = e['sections']
textstore = []
for i in range(len(sectionDict)):
    dictionary = sectionDict[i]
    subdictionary = dictionary['paragraphs']
    if len(subdictionary) > 1:
        for j in range(len(subdictionary)):
            sub = subdictionary[j]
            page = sub['page']
            if page == 0 or 1:          #The identifier sentence will only be on the first or second page of the article if it exists
                text = sub['text']
                textstore.append(text)
    if len(subdictionary) == 1:
        subdictionary = dictionary['paragraphs'][0]
        page = subdictionary['page']
        if page == 0 or 1:          #The identifier sentence will only be on the first or second page of the article if it exists
            text = subdictionary['text']
            textstore.append(text)
    if len(subdictionary) == 0:
        subdict = sectionDict[i]
        paradict = subdict['paragraphs']
        for j in range(0, len(paradict)):
            finaldict = paradict[j]
            page = finaldict['page']
            if page == 0 or 1:
                text = finaldict['text']
                textstore.append(text)

sentences = []
for para in textstore:
    s = split_into_sentences(para)
    for sent in s:
        sent = sent.lower()                                         #Make everything lower case
        sent = re.sub('['+string.punctuation+']', '', sent)
        words = sent.split()
        sentences.append(words)
sentences = np.array(sentences)
print (sentences.shape)

#Build the classifier, just using the most successful one from the analysis earlier, ExtraTreesClassifier with TfidfVectorizer
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
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesRegressor(n_estimators=220))])

all_models = [
    ('etree_w2v_tfidf', etree_w2v_tfidf)
]

print ('Benchmarking')
unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))

fittedModel = etree_w2v_tfidf.fit(X, y)
pred = fittedModel.predict(sentences)

flag = 0
s = []
p = []

for i in range(sentences.shape[0]):
    if pred[i] > 0.9:
        s.append(sentences[i])
        p.append(pred[i])
        flag = 1
if flag == 0:
    for i in range(sentences.shape[0]):
        if pred[i] > 0.8:
            s.append(sentences[i])
            p.append(pred[i])
            flag = 1
if flag == 0:
    print ('We dont think there is an indicator sentence')
    for i in range(sentences.shape[0]):
        if pred[i] > 0.7:
            s.append(sentences[i])
            p.append(pred[i])
            flag = 1
if flag == 0:
    print ("Couldn't find [an idenitifer sentence")

s = np.array(s)
p = np.array(p)
I = np.argmax(p)

print ('#########################################################')
print ('# ~~~~~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~~~ #')
print ('Identifier Found:')
#Stitch the sentence back together
identifier = s[I]
s = " "         #Use space as the thing to knit the sentences back together
identifier = s.join( identifier)
print (identifier)

familyFlag, i = family_check(identifier)
if familyFlag == 1:
    print ('We think that there is a family connection because we found:')
    print (i)

unusualFlag, i = unusual_check(identifier)
if unusualFlag == 1:
    print ('We think that this condition may be atypical because we found:')
    print (i)

number = number_of_patients(identifier)
if number != 0 or None:
    print ('We think that there are', number, 'patients in the article, because the largest number we found was:')
    print (number)
