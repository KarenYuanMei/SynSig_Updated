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

def compare_regressor_time():
	df=pd.read_csv('../run_ML/ML_output/regressors/regressor_time.csv', sep=',')
	regressors=df['regressor'].tolist()
	time=df['time'].tolist()
	reg_time=list(zip(regressors, time))
	d=defaultdict(list)
	for regressor, time in reg_time:
		d[regressor].append(time)
	
	mean_times=[]
	time_sems=[]
	for key in d:
		times=d[key]
		mean_time=np.mean(times)
		mean_times.append(mean_time)
		time_sem=stats.sem(times)
		time_sems.append(time_sem)

	return mean_times, time_sems

def calc_sweep_param(param_name, param_values):
	mean_aucs=[]
	for item in param_values:
		filename='../run_ML/ML_output/sweep_param_results/sweep_rf_%s_%s'%(param_name,item)
		tprs, mean_fpr, auc_list=find_5fold_roc(filename)
		print (item, np.mean(auc_list))
		mean_aucs.append(np.mean(auc_list))
		#graph_functions.plot_mean_ROC(tprs, mean_fpr, auc_list)
	return mean_aucs

if __name__ == '__main__':

	# regressors, mean_aucs, reg_sem=compare_regressor_roc()
	# labels=['Adaboost',  'Random Forest', 'SVM (poly deg=3)', 'SVM (poly deg=4)', 'SVM (rbf)', 'SVM (Sigmoid)']
	# graph_functions.plot_bargraph_with_errorbar(labels, mean_aucs, reg_sem, 'Cross-Validation', 'Synapse Recovery ROC', 'compare_regressor_roc')

	# mean_times, time_sems=compare_regressor_time()
	# graph_functions.plot_bargraph_with_errorbar(labels, mean_times, time_sems, 'Cross-Validation', 'Time(s)', 'regressor_time')

	tree_no=[100, 125, 150, 175, 200, 225, 250, 275, 300]
	tree_mean_aucs=calc_sweep_param('tree', tree_no)
	graph_functions.plot_linegraph(tree_no, tree_mean_aucs, 'Number of Trees', 'Synapse Recovery ROC', 'tree_no')

	max_depth=[10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
	depth_mean_aucs=calc_sweep_param('max_depth', max_depth)
	graph_functions.plot_linegraph(max_depth, depth_mean_aucs, 'Maximum Tree Depth', 'Synapse Recovery ROC', 'depth')

	min_samples_split=[2, 3, 4, 5, 10]
	split_mean_aucs=calc_sweep_param('split', min_samples_split)
	graph_functions.plot_linegraph(min_samples_split, split_mean_aucs, 'Minimum Samples Split', 'Synapse Recovery ROC', 'split')
