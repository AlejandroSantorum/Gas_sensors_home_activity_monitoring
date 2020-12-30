import math
import time
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix


def metric_report(y_true, y_pred, plot_conf_mtx=False, clf=None, X=None):
    '''
        INPUT:
            y_true: target true values
            y_pred: predicted classes for test data
            plot_conf_mtx: Boolean. If set to True, clf and X params are required and it prints
                           confusion matrix. Default: False
            clf: estimator used to predict y_pred values. It's required to plot conf matrix
            X: test inoput values used to predict y_pred using clf. It's required to plot conf matrix

        OUTPUT:
            None

        DESCRIPTION:
            It prints a whole metric/score report, focusing on accuracy, each class recall and f1-score.

    '''
    occurrences_dict = Counter(y_true)
    bg_perc = occurrences_dict['background']/len(y_true)
    bg_banana = occurrences_dict['banana']/len(y_true)
    bg_wine = occurrences_dict['wine']/len(y_true)

    conf_m = confusion_matrix(y_true, y_pred, labels=['background', 'banana', 'wine'])

    recall_bg = conf_m[0][0]/sum(conf_m[:,0])   # predicted_bg / total_real_bg
    recall_ban = conf_m[1][1]/sum(conf_m[:,1])   # predicted_banana / total_real_banana
    recall_wine = conf_m[2][2]/sum(conf_m[:,2]) # predicted_wine / total_real_wine

    print('-------------------------------------------')
    print('Real background percentage:', bg_perc)
    print('Real banana percentage:', bg_banana)
    print('Real wine percentage:', bg_wine)
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Recall on background:', recall_bg)
    print('Recall on banana:', recall_ban)
    print('Recall on wine:', recall_wine)
    print('F1-score:', f1_score(y_true, y_pred, average='weighted'))
    if plot_conf_mtx:
        if not (clf and X):
            raise ValueError('To plot confusion matrix is required estimator and input X')
        else:
            plot_confusion_matrix(clf, X, y_true)
    print('-------------------------------------------')





    


