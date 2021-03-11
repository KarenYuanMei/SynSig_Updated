import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import time
import scipy.stats as stats
import sklearn.metrics as metrics

import net_random_walk_functions

#calculate 5fold crossvalidation


def find_scores_df(df):
	cols=df.columns
	to_delete=[]
	for item in cols:
		if "Prop" not in item:
			to_delete.append(item)
	final=df.drop(to_delete, axis=1)
	return final

def find_val_df(df, neg_eval):
	pos=df[df['Non-Sample']==1]
	neg=df.loc[neg_eval]
	final=pd.concat([pos, neg])
	return final

def calculate_roc(df, neg_eval):
	final=find_val_df(df, neg_eval)
	#print (final)
	y_test=final['Non-Sample'].tolist()
	final=find_scores_df(final)
	#print (final)
	final['mean']=final.mean(axis=1)
	# calculate the fpr and tpr for all thresholds of the classification
	preds=final['mean'].tolist()
	fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
	roc_auc = metrics.auc(fpr, tpr)
	return fpr, tpr, threshold, roc_auc

def calc_cv_prop_aucs(df, neg):
	mean_fpr = np.linspace(0, 1, 100)
	tprs = []
	aucs = []
	for j in np.arange(0,14,3):
		#=print ('j', j)
		subdf=df.iloc[:, j:j+3]
		subdf.columns=['Sub-Sample', 'Non-Sample', 'Prop Score']
		fpr, tpr, threshold, roc_auc=calculate_roc(subdf, neg)
		final=pd.DataFrame({'Threshold': threshold, 'TPR': tpr, 'FPR': fpr})
		#print (df)
		#final.to_csv('../propagate_synapse/results/ROC_df_%s.csv'%j)
		#print ('actual', roc_auc)
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.
		aucs.append(roc_auc)
	return mean_fpr, tprs, aucs

#-----calculate net test results:
def calc_net_test_roc(df, neg):
	cols=['Sub-Sample', 'Non-Sample', 'Prop Score']
	subdf=df[cols]
	fpr, tpr, threshold, roc_auc=calculate_roc(subdf, neg)
	return fpr, tpr, threshold, roc_auc


def find_shuff_aucs(G, nodesets, neg, alpha, fraction, iterations):
	shuff_scores=[]
	for i in range(iterations):
		shuff=net_random_walk_functions.find_shuff_scores_df(G, nodesets, alpha, fraction)
		fpr, tpr, threshold, roc_auc=calc_net_test_roc(shuff, neg)
		shuff_scores.append(roc_auc)
	return shuff_scores

#sweep alpha:

def sweep_alpha_aucs(G, nodesets, neg):
	alphas=np.arange(0.1, 1, 0.1)

	all_mean_aucs=[]
	for item in alphas:
		kernel=net_random_walk_functions.construct_prop_kernel(G, item, verbose=True)
		df=net_random_walk_functions.find_prop_scores_df(kernel, nodesets, 0.8)
		#print (df)
		mean_fpr, tprs, aucs=calc_prop_aucs(df, neg)
		mean_aucs=np.mean(aucs)
		all_mean_aucs.append(mean_aucs)
	return all_mean_aucs
