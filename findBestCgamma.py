import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, mean_squared_error,accuracy_score
from scipy import interp
import random

import matplotlib.pyplot as plt

data = pd.read_csv('twogaussians.csv',header=None)
#data = pd.read_csv('twospirals.csv',header=None)
#data = pd.read_csv('halfkernel.csv',header=None)
#data = pd.read_csv('clusterincluster.csv',header=None)

data.columns = ['a','b','class']

#random.shuffle(data)
data = shuffle(data)
X = np.array(data.drop(['class'],1))
X = preprocessing.scale(X)
Y = np.array(data['class'])
    
for c in range(len(Y)):
    if(Y[c] == 1):
        Y[c] = 0
    else : Y[c] = 1
    
#random.shuffle(zip(X,Y))

C_range = np.logspace(-2, 10, 5)

gamma_range = np.logspace(-9, 3, 5)
degree = np.arange(2,11,1)
fpv = []
tprs = []
aucs = []
mse = []
lowest_mse = []
mean_fpr = np.linspace(0, 1, 100)
classifiers = []
acc = []

X_train = X[:-100]
Y_train = Y[:-100]
X_test = X[-100:]
Y_test = Y[-100:]

for C in C_range:
    for gamma in gamma_range:
        clf = svm.SVC(kernel='rbf',C=C, gamma=gamma,probability=True)
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        mse.append((mean_squared_error(Y_test, prediction),C,gamma))
        prediction = clf.predict(X_test)
        acc.append(accuracy_score(Y_test, prediction))
        probas_ = clf.predict_proba(X_test)
        # Compute ROC curve and area underthe curve
        
lowest_mse.append(min(mse))
lowest_mse.append(mse[3])
lowest_mse.append(mse[20])

for i in range(len(lowest_mse)):
    clf = svm.SVC(kernel='rbf',C=lowest_mse[i][1],
                  gamma=lowest_mse[i][2],probability=True)
    clf.fit(X_train, Y_train)
    probas_ = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
    #tprs.append(interp(mean_fpr, fpr, tpr))
    #tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.8,
                 label='C = %f gamma = %f (AUC = %0.2f)' %(lowest_mse[i][1],lowest_mse[i][2],roc_auc))
    
 
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and AUC for C and gamma')
plt.legend(loc="lower right")
plt.show()


print(min(mse))   
        




