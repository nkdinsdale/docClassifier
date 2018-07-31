#Import all the dependencies we know we are going to have
import numpy as np
import matplotlib.pyplot as plt
import re
from pprint import pprint
import string
from sklearn import svm
import sqlquery
import csv
from sklearn.feature_extraction.text import CountVectorizer
import svmFunction

#Read in all the titles
titles = []
useful = []
year = []
doi = []
disorderFound = []
disorderSearch = []
## Create the input data ##
#Read in the positive data set
with open('pos_titles.csv', 'r') as f:
    reader = csv.reader(f, delimiter = ',')
    i = 0       #Create an iterator to remove the headers
    for row in reader:
        if i == 0:
            #The first row contains the column headers so dont want to include these
            headers = row
        else:
            title = row[1]
            #strip out all the punctuation from the titles
            title = re.sub('['+string.punctuation+']', '', title)
            titles.append(title)
            useful.append(row[2])
            year.append(row[3])
            doi.append(row[4])
            disorderFound.append(row[5])
            disorderSearch.append(row[6])
        i += 1

posTotal = i        #There are way more negatives than positives so cant use all the negatives
#now read in the negative data set
with open('neg_titles.csv', 'r') as f:
    reader = csv.reader(f, delimiter = ',')
    i = 0   #Create an iterator to remove the headers
    for row in reader:
        if i == 0:
            #The first row contains the column headers so dont want to include these
            headers = row
        else:
            if i < posTotal :            #So have the same number of positive and negative titles for now
                title = row[1]
                #strip out all the punctuation from the titles
                title = re.sub('['+string.punctuation+']', '', title)
                titles.append(title)
                useful.append(row[2])
                year.append(row[3])
                doi.append(row[4])
                disorderFound.append(row[5])
                disorderSearch.append(row[6])
        i += 1

titles = np.array(titles)

#Read in the stop words
with open('stopwords.txt', 'r') as file:
    stopWords =  file.read().splitlines()

#Get rid of the punctuation and stop words to make the bag of words analysis easier
wordStore = []
syndromeWord = np.zeros(len(useful))
i = 0
for title in titles:
    #strip out the punctuation
    title = re.sub('['+string.punctuation+']', '', title)
    #make everything lower case --> case doesnt matter
    title = title.lower()
    #Also remove the word if its the name the syndrome
    idnumber = disorderSearch[i]
    syndromes = sqlquery.find_syndrome(idnumber)
    #syndromes = syndromes.split()
    for syn in syndromes:
        syn = syn.lower()
        id = re.finditer(syn, title, re.MULTILINE | re.IGNORECASE)
        for match in id:
            syndromeWord[i] = 1
            flag = 1
            text = match.group(0)
            title = re.sub(text, '', title)
    #Split into words and get rid of the stop words
    words = title.split()
    for word in words:
        flag = 0
        #Remove the word if its a stop word
        for stop in stopWords:
            id = re.match(stop, word, re.MULTILINE | re.IGNORECASE)
            if id != None:
                flag = 1
        if flag == 0:
            wordStore.append(word)
    i += 1

#find the unique data set --> this will be the feature vector for all possible words
used = set()        #Create a set to use to find all the unique words
unique = [x for x in wordStore if x not in used and (used.add(x) or True)]
unique = np.array(unique)

#create the feature vector based of this, also include the data and DOI number like before
#the truth is still given by useful
useful = np.array(useful).astype(float)    #Is the y vector, swap the strings to ints
#Create an empty feature vector to update
features = np.zeros((len(useful), len(unique)+3))

#First feature is whether the DOI was found or not
features[:,0] = np.array(doi).astype(int)

#Second feature is the age of the article
#1960 = 0, 2018 = 1
#Assuming older articles are less likely to be useful for instance because of changes in how articles are named
year = np.array(year).astype(int)
year = (year - 1960)/(2018 - 1960)          #Normalise the year
features[:,1] = year

#Third feature is whether the title name contains the syndrome name
syndromeWord = np.array(syndromeWord).astype(int)
features[:,2] = syndromeWord

#All the rest of the features vector will be the bag of words with the stop words removed
#Obviously creates a really sparse feature vector
i = 0           #So we know which title we are on
for title in titles:
    col = 3                                                 #Columns start at three because we already have the first three hardcoded
    title = re.sub('['+string.punctuation+']', '', title)   #Remove the punctuation
    title = title.lower()                                   #Make everything lower case
    for word in unique:
        sum = title.count(word)     #count how many times the word appears
        features[i,col] = sum
        col += 1
    i += 1


pthreshold = []
rthreshold = []
athreshold = []
thresholds = np.linspace(0,1,75)
for threshold in thresholds:
    p = []
    r = []
    a = []
    for run in range(0,5):
        #Split into testing and training examples
        shuffler = np.random.permutation(range(features.shape[0]))
        features = features[shuffler,:]
        useful = useful[shuffler]
        proportion = int(0.7 * features.shape[0])
        Xtest = features[proportion:,:]
        Xtrain = features[0:proportion,:]
        ytest = useful[proportion:]
        ytrain = useful[0:proportion]

        #Create the support vector machine
        clf = svm.SVR()
        fit = clf.fit(Xtrain, ytrain)
        predictions = clf.predict(Xtest)

        print('Predictions:')
        print (predictions)

        predictions[predictions > threshold ] = 1
        predictions[predictions != 1 ] = 0

        print ('Truths:')
        print (ytest)

        accuracy, recall, precision = svmFunction.make_stats(predictions, ytest)
        print ('Accuracy = ', accuracy)
        print ('Recall = ', recall)
        print ('Precision =', precision)
        p.append(precision)
        r.append(recall)
        a.append(accuracy)
    print ('Average Results:')
    print ('Precision:', np.mean(p), np.std(p))
    print ('Recall:', np.mean(r), np.std(r))
    print ('Accuracy:', np.mean(a), np.std(a))
    pthreshold.append(np.mean(p))
    rthreshold.append(np.mean(r))
    athreshold.append(np.mean(a))

plt.figure()
plt.plot(np.transpose(thresholds), pthreshold, '-', linewidth=2, label='Precision')
plt.plot(np.transpose(thresholds), rthreshold, '-', linewidth=2, label='Recall')
plt.plot(np.transpose(thresholds), athreshold, '-', linewidth=2, label='Accuracy')
plt.xlabel('Threshold')
plt.legend(['Precision', 'Recall', 'Accuracy'])
plt.show()

plt.figure()
plt.plot(rthreshold, pthreshold, 'x', linewidth =2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
