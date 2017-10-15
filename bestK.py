#Author: Zarreen Naowal Reza
#Email: zarreen.naowal.reza@gmail.com

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

#data = pd.read_csv('twogaussians.csv',header=None)
data = pd.read_csv('twospirals.csv',header=None)
#data = pd.read_csv('halfkernel.csv',header=None)
#data = pd.read_csv('clusterincluster.csv',header=None)

print("classifier: k-Nearest Neighbor\n")

data.columns = ['a','b','class']

def clf_eval(Y_test, prediction):
      
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(Y_test)):
        if Y_test[i] == prediction[i] == 1:
            TP += 1
    for i in range(len(Y_test)):
        if Y_test[i] == prediction[i] == 2:
            TN += 1
    for i in range(len(Y_test)):
        if Y_test[i] == 2 and prediction[i] == 1:
            FP += 1
    for i in range(len(Y_test)):
        if Y_test[i] == 1 and prediction[i] == 2:
            FN += 1
                
    return TP,TN,FP,FN

X = np.array(data.drop(['class'],1))
Y = np.array(data['class'])
X = preprocessing.scale(X)

#1 = positive, 2=negative
accuracy = []
ppv = []
npv = []
specificity = []
sensitivity = []

all_ppv = [[]]
all_npv = [[]]
all_specificity = [[]]
all_sensitivity = [[]]
all_accuracy = [[]]

k_range = int(np.sqrt(len(data)))+1
for n in range(1,k_range):
    
    kf = KFold(n_splits=10,shuffle=True)
    for train_index, test_index in kf.split(X):
           
        X_train, X_test = X[train_index],X[test_index]
        Y_train, Y_test = Y[train_index],Y[test_index]
            
        classifier = KNeighborsClassifier(n_neighbors=n,p=2,n_jobs=-1)
        classifier.fit(X_train,Y_train)
        prediction = classifier.predict(X_test)
        TP,TN,FP,FN = clf_eval(Y_test, prediction)
        ppv.append(TP/(TP+FP))
        npv.append(TN/(TN+FN))
        specificity.append(TN/(TN+FP))
        sensitivity.append(TP/(TP+FN))
        accuracy.append((TP+TN)/len(Y_test))
        
        
    all_ppv.append([np.mean(ppv),n])
    all_npv.append([np.mean(npv),n])
    all_specificity.append([np.mean(specificity),n])
    all_sensitivity.append([np.mean(sensitivity),n])
    all_accuracy.append([np.mean(accuracy),n])
    
print("max ppv: ",max(all_ppv))
print("max npv: ",max(all_npv))
print("max specificity: ",max(all_specificity))
print("max sensitivity: ",max(all_sensitivity))
print("max accuracy: ",max(all_accuracy))

        

