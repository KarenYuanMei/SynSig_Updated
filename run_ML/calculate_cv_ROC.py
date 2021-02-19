#GOal: 1) To calculate the 5fold crossvaliation ROC 
#      2) Plot ROC

import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 14})


from sklearn.metrics import auc
#from itertools import combinations
from collections import defaultdict

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from scipy import stats

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
	#print (auc_list)
	return tprs, mean_fpr, auc_list

def compare_regressor_roc():
	regressors=['adaboost', 'random_forest', 'svregressor_3', 'svregressor_4', 'svregressor_rbf', 'svregressor_sigmoid']
	mean_aucs=[]
	reg_sem=[]
	for item in regressors:
		filename='../run_ML/ML_output/regressors/full60_'+item
		tprs, mean_fpr, auc_list=find_5fold_roc(filename)
		mean_auc=np.mean(auc_list)
		sem=stats.sem(auc_list)
		mean_aucs.append(mean_auc)
		reg_sem.append(sem)
	return regressors, mean_aucs, reg_sem



if __name__ == '__main__':
	# filename='../run_ML/sweep_param_results/sweep_rf_tree_100'
	# tprs, mean_fpr, auc_list=find_5fold_roc(filename)
	# print (np.mean(auc_list))
	# graph_functions.plot_mean_ROC(tprs, mean_fpr, auc_list)

	#regressors, mean_aucs, reg_sem=compare_regressor_roc()
	#labels=['Adaboost',  'Random Forest', 'SVM (poly deg=3)', 'SVM (poly deg=4)', 'SVM (rbf)', 'SVM (Sigmoid)']
	#graph_functions.plot_bargraph_with_errorbar(labels, mean_aucs, reg_sem, 'Cross-Validation', 'Synapse Recovery ROC')

	df=pd.read_csv('../run_ML/ML_output/regressors/regressor_time.txt', sep='\t')
	print (df)
	print (df[0])
	print (df[0][0])

