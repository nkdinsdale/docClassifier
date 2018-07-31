import numpy as np
import matplotlib.pyplot as plt
import csv
from pprint import pprint
import re
import string
from sklearn import svm
import sqlquery
import svmFunction

#SVM --> http://scikit-learn.org/stable/modules/svm.html
pthreshold = []
rthreshold = []
athreshold = []
thresholds = np.linspace(0,1.2,25)
thresholds = [0.7]
for threshold in thresholds:
    print (threshold)
    p = []
    r = []
    a = []
    for i in range(0, 5):
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
        print (len(titles))

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
                    if 150 < i < posTotal+150:
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

        #useful is the ground truth for training, currently in the form of string so convert to float because doing regression not classifcation
        useful = np.array(useful).astype(float)    #Is the Y vector

        #Create a numpy array of zeros to use as the feature vector
        features = np.zeros((useful.shape[0], 25))         #For now just make five features this will increase later

        #First feature is whether the DOI was found or not
        features[:,0] = np.array(doi).astype(int)

        #Second feature is the age of the article, 1960 = 0, 2018 = 1
        #Assuming older articles are less likely to be useful
        year = np.array(year).astype(int)
        year = (year - 1960)/(2018 - 1960)
        features[:,1] = year

        #Next features refer to whether certain words appear in the title, try: patient, syndrome, case, existance of a MIM number etc
        #Search for patient, syndrome, case
        titles = np.array(titles)
        i = 0
        #Make a list of the key words using as indicators
        indicators = ['[P,p]roteus', '[M,m]utation', '[F,f]acial', '[C,c]ranio', '[M,m]utations', '[P,p]henotype', '[T,t]ype', '[C,c]linical']
        for title in titles:
            col = 2 #first location for these is column 2
            for exp in indicators:
                id = re.finditer(exp, title, re.MULTILINE | re.IGNORECASE)
                for match in id:
                    features[i,col] = 1       #Set the feature vector to one if it includes the buzz words
                col += 1      #put the next one in the next column
            i += 1

        #Search for a MIM number
        i = 0
        #look for a five or six digit combination
        indicators = ['[A-Z|0-9][A-Z|0-9][A-Z|0-9][A-Z|0-9][A-Z|0-9][A-Z|0-9]', '[A-Z|0-9][A-Z|0-9][A-Z|0-9][A-Z|0-9][A-Z|0-9]']
        for title in titles:
            for exp in indicators:
                id = re.finditer(exp, title, re.MULTILINE)      #DONT WANT IGNORECASE THEY NEED TO BE UPPER CASE
                for match in id:
                    features[i,col] = 1
            i += 1
        col += 1 #Increment col so the next features go in the right place

        #Also have negative buzz words that we dont want it to contain
        i = 0
        indicators = ['[M,m]ouse', '[Z,z]ebrafish', '[D,d]rosophila', '[F,f]lies', '[F,f]eet', '[F,f]ragile', 'x', '[P,p]raderwilli', '[P,p]remutation', '[C,c]hildren', '[S,s]tudy']
        for title in titles:
            colstore = col
            for exp in indicators:
                id = re.finditer(exp, title, re.MULTILINE | re.IGNORECASE)
                for match in id:
                    features[i,colstore] = 1       #Set the feature vector to one if it includes the buzz words
                colstore += 1      #put the next one in the next column
            i += 1
        col = colstore

        #See if the title contains the syndrome name or any of its synonyms
        i = 0           #iterator is the index to find the id number
        col = col + 1
        for title in titles:
            number = disorderSearch[i]
            results = sqlquery.find_syndrome(number)
            for res in results:
                id = re.finditer(res, title, re.MULTILINE | re.IGNORECASE)
                for match in id:
                    features[i, col] = 1
            i += 1

        #for i in range (features.shape[1]):
        #    print (features[:,i])

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
        clf = svm.SVR()                             #SVR = support vector regression rather than classification
        fit = clf.fit(Xtrain, ytrain)
        predictions = clf.predict(Xtest)

        print('Predictions before thresholding:')
        print (predictions)

        #threshold = 0.5         #For now just set the threshold to be 0.5, play with this later
        predictions[predictions > threshold]  = 1
        predictions[predictions != 1] = 0

        print('Predictions after thresholding:')
        print (predictions)
        print ('Truths:')
        print (ytest)

        accuracy, recall, precision = svmFunction.make_stats(predictions, ytest)
        print ('Accuracy = ', accuracy)
        print ('Precision =', precision)
        print ('Recall = ', recall)

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
