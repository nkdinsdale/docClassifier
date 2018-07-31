import numpy as np

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
