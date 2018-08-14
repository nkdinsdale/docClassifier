# 2018 Nicola Dinsdale
# Dependencies: pdffigures2-assembly-0.0.12-SNAPSHOT.jar --> extracts the information from the pdf to a json file.
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
import subprocess
import sys
import PyPDF2
import pdfminer

#Project files
from Embeddings import TfidfEmbeddingVectorizer
from re_functions import *
from regressor_functions import *

def get_info(path):
    with open(path, 'rb') as f:
        pdf =PyPDF2.PdfFileReader(f)
        info = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()
    author = info.author
    creator = info.creator
    subject = info.subject
    title = info.title

    return info, number_of_pages

################################################################################
# Get the file name from the input argument
# Input parser
# THIS SHOULD BE IN THE FORM OF A PDF
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', required=True, help="path to f file, f should be a PDF")
args = parser.parse_args()
file_pth = args.file
print (file_pth)
assert os.path.isfile(file_pth)

#use pyPDF2 to get some information quickly
info, number_of_pages = get_info(file_pth)


#Run the jar to create the json files and extract the figures
print ('EXTRACTING THE BODY OF THE DOCUMENT')
x = subprocess.call(['java','-jar', 'pdffigures2-assembly-0.0.12-SNAPSHOT.jar', file_pth,'-g', 'g_'])
# We cannot run this extraction on the older pdfs so force an abort if we have not succeeded
if x == 1:
    print ('ERROR: FAILED TO EXTRACT INFORMATION FROM PDF')
    sys.exit()

print ('EXTRACTING THE IMAGES FROM THE DOCUMENT')
y = subprocess.call(['java', '-jar', 'pdffigures2-assembly-0.0.12-SNAPSHOT.jar', file_pth, '-m', 'm_'])
if y == 1:
    print('Failed to extract images')
no = int([float(s) for s in re.findall(r'-?\d+\.?\d*', file_pth)][0])     #Use re to extract the number
filename = 'g_id_article_' + str(no) + '.json'
#Read in the pdf structure in the form of a json --> dictionary of dictionaries
print ('Reading in file:')
with open(filename) as json_data:
    e = json.load(json_data)
    json_data.close()

#Search the sections to try and find the sentence, its only going to be on page one or two
#Some of the articles store the abstract as a seperate item so look for this and search it as well7
textstore = []
print ('SEARCHING FOR ABSTRACT')
try:
    Abstract = e['abstractText']
    print ('\t Abstract object found')
    print ('\t Gathering Text from abstract')
    textstore.append(Abstract['text'])
except:
    print ('\t No abstract object ')

print ('GATHERING MAIN TEXT')

sectionDict = e['sections']
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

fittedModel = idenitifier_regressor()
pred_id = fittedModel.predict(sentences)

#Do the same for the captions
caps = []
pageno = []
tableFlag = 0
figDict = e['figures']
for k in range(len(figDict)):
    fig = figDict[k]
    if fig['figType'] == 'Figure':
        caps.append(fig['caption'])
        pageno.append(fig['page'])
    if fig['figType'] == 'Table':
        tableFlag += 1
capStore = []
for cap in caps:
    cap = cap.lower()       #Make everything lower case
    cap = re.sub('['+string.punctuation+']', '', cap)
    words = cap.split()
    capStore.append(words)

capmodel = caption_regressor(verbose = 1)
pred_cap = capmodel.predict(capStore)
args = np.argsort(pred_cap)
args = args[::-1]

# Run the analysis given the model
flagid = 0
sid = []
pid = []

for i in range(sentences.shape[0]):
    if pred_id[i] > 0.9:
        sid.append(sentences[i])
        pid.append(pred_id[i])
        flagid = 1
if flagid == 0:
    for i in range(sentences.shape[0]):
        if pred_id[i] > 0.8:
            sid.append(sentences[i])
            pid.append(pred_id[i])
            flagid = 1
if flagid == 0:
    #print ('We dont think there is an indicator sentence')
    for i in range(sentences.shape[0]):
        if pred_id[i] > 0.7:
            sid.append(sentences[i])
            pid.append(pred_id[i])
            flagid = 1
if flagid == 0:
    print ("Couldn't find an idenitifer sentence")

I = np.argmax(pid)

print ('#########################################################')
print ('# ~~~~~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~~~ #')

print ('TITLE:')
print (info.title)
print ('\n')

if flagid != 0:
    print ('IDENTIFIER SENTENCE:')
    print ('Identifier Found:')
    #Stitch the sentence back together
    if pid[I] < 0.8:
        print ('We dont think there is an identifier sentence. This was the highest scorer:')
    identifier = sid[I]
    s = " "         #Use space as the thing to knit the sentences back together
    identifier = s.join(identifier)
    print ('\n',identifier)
    print ('This identifier scored: ', pid[I])

    familyFlag, i = family_check(identifier)
    if familyFlag == 1:
        print ('We think that there is a family connection because we found:')
        print (i)

    unusualFlag, i = unusual_check(identifier)
    if unusualFlag == 1:
        print ('We think that this condition may be atypical because we found:')
        print (i)


print ('\n \n FIGURE CAPTIONS:')
print ('\t WARNING: pdffigures2 doesnt always extract all the figures succesfully \n')
for a in args:
    print ('- We are %2f confident this figure has a face in it:' % pred_cap[a])
    c = capStore[a]
    s = " "         #Use space as the thing to knit the sentences back together
    c = s.join(c)
    print (c)
    print ('Found on page', pageno[a])
    print ('Searching for identifiers:')
    identifiers = indentifier_analyis(c)
    print (identifiers)
    familyFlag, i = family_check(c)
    if familyFlag == 1:
        print ('- We think that there is a family connection because we found:')
        print (i)
    unusualFlag, i = unusual_check(c)
    if unusualFlag == 1:
        print ('- We think that this condition may be atypical because we found:')
        print (i)
    surgery_flag, i = surgery(c)
    if surgery_flag == 1:
        print ('- WARNING: We think that the person may have had reconstructive surgery because we found:')
        print (i)
    flag, i = unaffectedCheck(c)
    if flag == 1:
        print ('- WARNING: We think that someone in the photo is not affected because we found:')
        print (i) 
    print ('\n')

if tableFlag != 0:
    print ('We think there is information held in a table in this article, we found ', tableFlag, ' tables')
