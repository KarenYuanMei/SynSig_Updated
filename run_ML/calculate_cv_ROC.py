#GOal: 1) To calculate the 5fold crossvaliation ROC 
#      2) Plot ROC

import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

from sklearn.metrics import auc
from itertools import combinations
from collections import defaultdict

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
matplotlib.rcParams.update({'font.size': 14})

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../graph_functions/')
import graph_functions

sys.path.append('../ML_functions/')
import ROC_functions


def find_5fold_roc(filename):
	
	mean_fpr = np.linspace(0, 1, 100)
	#print ('mean fpr', mean_fpr)
	tprs=[]
	auc_list=[]
	for i in range(5):
		tpr, fpr, thresholds, auc=ROC_functions.find_all_auc_tprs(filename,i)
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		auc_list.append(auc)

	print (auc_list)
	return tprs, mean_fpr, auc_list

if __name__ == '__main__':
	filename='../run_ML/sweep_param_results/sweep_rf_treeno_100'
	tprs, mean_fpr, auc_list=find_5fold_roc(filename)
	print (np.mean(auc_list))
	graph_functions.plot_mean_ROC(tprs, mean_fpr, auc_list)