import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

#data = pd.read_csv('twogaussians.csv',header=None)
#data = pd.read_csv('twospirals.csv',header=None)
#data = pd.read_csv('halfkernel.csv',header=None)
#data = pd.read_csv('clusterincluster.csv',header=None)

print("classifier: Polynomial kernel SVM\n")

datasets = ['twogaussians.csv','twospirals.csv','halfkernel.csv','clusterincluster.csv']
for i in range(len(datasets)):
    data = pd.read_csv(datasets[i],header=None)
    data.name = str(datasets[i])

    data.columns = ['a','b','class']
    
    #print(data.isnull().sum()) #checking if there is any missing value
    #print(data.head())

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
    kf = KFold(n_splits=10,shuffle=True)
    for train_index, test_index in kf.split(X):
       
        X_train, X_test = X[train_index],X[test_index]
        Y_train, Y_test = Y[train_index],Y[test_index]
        
        classifier = svm.SVC(kernel="poly",degree=2)
        classifier.fit(X_train,Y_train)
        prediction = classifier.predict(X_test)
        TP,TN,FP,FN = clf_eval(Y_test, prediction)
        try:
            ppv.append(TP/(TP+FP))
        except ZeroDivisionError:
            ppv.append(0)
        try:
            npv.append(TN/(TN+FN))
        except ZeroDivisionError:
            npv.append(0)
        specificity.append(TN/(TN+FP))
        sensitivity.append(TP/(TP+FN))
        accuracy.append((TP+TN)/len(Y_test))
    
    print("dataset: ",data.name)
    print("mean ppv: ",np.mean(ppv))
    print("mean npv: ",np.mean(npv))
    print("mean specificity: ",np.mean(specificity))
    print("mean sensitivity: ",np.mean(sensitivity))
    print("mean accuracy: ",np.mean(accuracy))
    print('\n')

    #Plotting the samples
##    colors = ['w','b','r']
##    markers = ['*','x','o']
##    for i in range(len(X)):
##        plt.scatter(X[i][0],X[i][1],c = colors[Y[i]],marker= markers[Y[i]] )
##    plt.show()
        

