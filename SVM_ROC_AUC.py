import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, mean_squared_error
from scipy import interp

import matplotlib.pyplot as plt

print("classifier: RBF kernel SVM\n")

datasets = ['twogaussians.csv','twospirals.csv','halfkernel.csv','clusterincluster.csv']
for i in range(len(datasets)):
    data = pd.read_csv(datasets[i],header=None)
    data.name = str(datasets[i])

    data.columns = ['a','b','class']
    X = np.array(data.drop(['class'],1))
    X = preprocessing.scale(X)
    Y = np.array(data['class'])
    
    for c in range(len(Y)):
            if(Y[c] == 1):
                Y[c] = 0
            else : Y[c] = 1
    
    kf = KFold(n_splits=10,shuffle=True)
    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    fpv = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
##    C_2d_range = [1e-2, 1, 1e2]
##    gamma_2d_range = [1e-1, 1, 1e1]
    for train, test in kf.split(X):
        classifier = svm.SVC(kernel='rbf', probability=True,C=10000.0,
                             gamma=1.0, random_state=None,)
        probas_ = classifier.fit(X[train], Y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        #tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.8,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve and AUC of %s'%data.name)
    plt.legend(loc="lower right")
    plt.show()
    
    #Plotting the samples
##    colors = ['w','b','r']
##    markers = ['*','x','o']
##    for i in range(len(X)):
##        plt.scatter(X[i][0],X[i][1],c = colors[Y[i]],marker= markers[Y[i]] )
##    plt.show()
        

