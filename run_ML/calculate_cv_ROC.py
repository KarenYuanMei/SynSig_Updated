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

def load_resource_genes(i):
	test_genes_file='../updated_test_genes_%s.csv'%i
	test_genes=load_data_functions.get_gene_names(test_genes_file)

	training_genes_file='../updated_training_genes_%s.csv'%i
	training_genes=load_data_functions.get_gene_names(training_genes_file)

	positive_filename='../updated_positives.csv'
	positives=load_data_functions.get_gene_names(positive_filename)

	negative_filename='../updated_negatives.csv'
	negatives=load_data_functions.get_gene_names(negative_filename)
	return training_genes, test_genes, positives, negatives

def make_score_table(data, training_positives):
	gene1=data['Gene1'].tolist()
	gene2=data['Gene2'].tolist()
	df=data[['Gene2', 'Gene1', 'ytest', 'ypredict']]
	table = pd.pivot_table(df, values='ypredict', index=['Gene2'], columns=['Gene1'], aggfunc=np.sum)
	idx = table.index[table.isnull().all(1)]
	table=table[training_positives]
	table['mean']=table.mean(axis=1)
	print (table)
	return table

def find_group_df(table, genelist, test, label):
	group_test=list(set(genelist)&set(test))
	group_df=table.loc[group_test]
	group_df['group']=label
	return group_df

def make_input_ROC_df(data, training_positives, pos, neg, test):
	table=make_score_table(data, training_positives)
	pos_df=find_group_df(table, pos, test, 1)
	neg_df=find_group_df(table, neg, test, 0)
	input_df=pd.concat([pos_df, neg_df])
	print ('input', input_df)
	avg_score=input_df['mean'].tolist()
	label=input_df['group'].tolist()
	return label, avg_score

def save_roc_df(thresholds, tpr, fpr, i):
	ROC=list(zip(thresholds, tpr, fpr))
	ROC_df=pd.DataFrame({'Threshold': thresholds, "True_Positives": tpr, "False_Positives": fpr})
	ROC_df.to_csv('cv_roc_df_%s.csv'%i)
	return ROC_df

def calculate_roc(label, avg_score):
	fpr, tpr, thresholds = roc_curve(label, avg_score)
	auc = roc_auc_score(label, avg_score)
	return fpr, tpr, thresholds, auc

def find_all_auc_tprs(filename,i):
	data=pd.read_csv(filename +'_%s.csv'%i)
	training, test, pos, neg=load_resource_genes(i)
	training_positives=list(set(training)&set(pos))
	label, avg_score=make_input_ROC_df(data, training_positives, pos, neg, test)
	fpr, tpr, thresholds, auc=calculate_roc(label, avg_score)
	ROC_df=save_roc_df(thresholds, tpr, fpr)
	return tpr, fpr, thresholds, auc


filename='sweep_rf_treeno_100'
tprs=[]
auc_list=[]
for i in range(5):
	tpr, fpr, thresholds, auc=find_all_auc_tprs(filename,i)
	tprs.append(tpr)
	auc_list.append(auc)

print (tprs)
print (auc_list)

