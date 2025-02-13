#Goal: to show how predictions compare to housekeeping and other gene lists
#run in svm-poly environment
# Figure 2C

import pandas as pd
import networkx as nx
import numpy as np
from numpy import random

from mlxtend.evaluate import permutation_test
from scipy import stats

import ddot
from ddot import Ontology
import csv

import random
from statistics import mean
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})

from itertools import combinations, combinations_with_replacement
from itertools import product
from collections import defaultdict

from sklearn.metrics import roc_auc_score

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import define_gene_objects
#import regressor_functions
import find_training_genes_functions
import find_GO_scores
#import run_train_crossvalidate_pipeline
#import define_features
import ROC_functions
#import graph_functions

import sys
sys.path.append('../graph_functions/')
import graph_functions

#find housekeeping:==============
#housekeeping genes from paper: Hounkpe et al, 2021
def find_hk(big_pool):
	hk=pd.read_csv('../source_data_files/gene_lists/Human_Mouse_Common.csv', sep=';')
	#print (hk)
	hk=hk['Human'].tolist()
	hk=list(set(hk)&set(big_pool))
	return hk

#find nuclear proteins:=========
#downloaded from https://www.proteinatlas.org/search/subcell_location%3ANucleoplasm%2CNuclear+speckles%2CNuclear+bodies%2CKinetochore%2CMitotic+chromosome, 06/24/21
def find_nuclear(big_pool):
	nuclear=pd.read_table('../source_data_files/gene_lists/subcell_location_Nucleoplasm_Nuclear.tsv')
	print (nuclear)
	nuclear=nuclear['Gene'].tolist()
	print (nuclear[:5])
	nuclear=list(set(nuclear)&set(big_pool))
	return nuclear

#find golgi (Golgi):===================
def find_golgi(big_pool):
	golgi=load_data_functions.get_gene_names('../source_data_files/gene_lists/golgi_genes.csv')
	golgi=list(set(golgi)&set(big_pool))
	return golgi

#find transmembrane:(uniprot)==============
def find_mem(big_pool):
	transm=pd.read_csv('../source_data_files/gene_lists/Uniprot_transmembrane.csv')
	transm=transm['Gene names'].tolist()
	transm=[str(x) for x in transm]
	mem=[]
	for item in transm:
		entry=item[:item.find(' ')]
		mem.append(entry)
	mem=list(set(mem)&set(big_pool))
	return mem

#load the synapse gene lists and the control gene lists:
def load_control_and_synapse_genes(big_pool, go_genes):
	hk=find_hk(big_pool)
	nuclear=find_nuclear(big_pool)
	golgi=find_golgi(big_pool)
	mem=find_mem(big_pool)
	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	syndb=load_data_functions.find_SynDB(big_pool)
	synsysnet=load_data_functions.find_synsysnet(big_pool)
	return hk, nuclear, golgi, mem, syngo, syndb, synsysnet

def compare_auc_bootstrap(set1_predictions,set2_predictions):
	#set1_predictions and set2_predictions should be the output from find_true_y
	#returns a confidence interval for the difference between the auc scores for the two sets
	scores = set1_predictions['avg_scores']

	set1_labels = set1_predictions['label']
	set2_labels = set2_predictions['label']

	num_bootstrap_samples = 10000

	bootstrapped_auc_diffs = []
	for i in range(num_bootstrap_samples):
		#indices = random.randint(0,len(scores))
		indices=list(np.random.randint(low = 0,high=len(scores),size=len(scores)))
		#print (indices)

		set1_auc = roc_auc_score(set1_labels[indices],scores[indices])
		set2_auc = roc_auc_score(set2_labels[indices],scores[indices])

		diff = set1_auc - set2_auc

		bootstrapped_auc_diffs.append(diff) 


	conf_interval_sizes = [0.95,0.99, 0.999]
	conf_intervals = {}

	bootstrapped_auc_diffs.sort()

	for interval_size in conf_interval_sizes:
		#print (interval_size)
		lower_bound_index = int(num_bootstrap_samples*((1-interval_size)/2))
		#print (lower_bound_index)

		lower_bound = bootstrapped_auc_diffs[lower_bound_index]

		upper_bound_index = int(num_bootstrap_samples*(interval_size+((1-interval_size)/2)))
		#print (upper_bound_index)
		upper_bound = bootstrapped_auc_diffs[upper_bound_index]

		conf_intervals[interval_size] = (lower_bound,upper_bound)

	return conf_intervals


def auc_bootstrap_errorbars(set1_predictions):
	#set1_predictions should be the output from find_true_y
	#returns a confidence interval for the auc score for the set
	scores = set1_predictions['avg_scores']

	set1_labels = set1_predictions['label']

	num_bootstrap_samples = 1000

	bootstrapped_auc= []
	for i in range(num_bootstrap_samples):
		#indices = random.randint(0,len(scores))
		indices=list(np.random.randint(low = 0,high=len(scores),size=len(scores)))
		#print (indices)

		set1_auc = roc_auc_score(set1_labels[indices],scores[indices])


		bootstrapped_auc.append(set1_auc) 


	conf_interval_sizes = [0.95]
	conf_intervals = {}

	bootstrapped_auc.sort()

	for interval_size in conf_interval_sizes:
		#print (interval_size)
		lower_bound_index = int(num_bootstrap_samples*((1-interval_size)/2))
		#print (lower_bound_index)

		lower_bound = bootstrapped_auc[lower_bound_index]

		upper_bound_index = int(num_bootstrap_samples*(interval_size+((1-interval_size)/2)))
		#print (upper_bound_index)
		upper_bound = bootstrapped_auc[upper_bound_index]

		conf_intervals[interval_size] = (lower_bound,upper_bound)

		errorbars=(upper_bound-lower_bound)/2

	return conf_intervals, errorbars


#find the roc aucs of the predicted scores against the reference genelists
def compute_pred_dfs_aucs(genelists, all_training):
	pred_dfs=[]
	aucs=[]
	predicted=load_data_functions.load_predicted_synsig_df()
	for item in genelists:
		pred_df, labels, avg_scores=ROC_functions.find_pred_labels_scores(predicted, item, all_training)
		#print (final)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(labels, avg_scores)
		aucs.append(auc)
		#print (auc)
		pred_dfs.append(pred_df)
	return pred_dfs, aucs

#calculate the confidence intervals for the synapse control genes
def compute_syn_control_ci(genelists, genelist_names, final_dfs):
	genelist_diff_ci={}
	for i in range(1, len(genelists)):
		conf_interval=compare_auc_bootstrap(final_dfs[0], final_dfs[i])
		#print (conf_interval)
		genelist_diff_ci[genelist_names[i]]=conf_interval
	return genelist_diff_ci


#calculate the true positive rate (tpr) and false positive rate(fpr):
def calc_syn_tpr_fpr(ref_list, genelist, big_pool, all_training):
	ref_list=list(set(ref_list)-set(all_training))
	genelist=list(set(genelist)-set(all_training))
	tp=ref_list
	found_pos=list(set(tp)&set(genelist))
	tpr=float(len(found_pos)/len(tp))

	fp=list(set(genelist)-set(ref_list))
	tn=list(set(big_pool)-set(ref_list))
	fpr=float(len(fp)/(len(fp)+len(tn)))
	return tpr, fpr

#calculate the true positive rate (tpr) and false positive rate (fpr) for the control genesets against the synapse genes:
def calc_ctrl_tpr_fpr(syn, big_pool, all_training):

	ratios={}
	controls=[hk, nuclear, golgi, mem]
	control_names=['hk', 'nuclear', 'golgi', 'mem']
	for i in range(len(controls)):
		tpr, fpr=calc_syn_tpr_fpr(syn, controls[i], big_pool, all_training)
		print (tpr, fpr)

		ratios[control_names[i]]=(tpr, fpr)
	return ratios

#plot the roc recovery of each geneset being used as the reference:
def plot_controls_with_errorbar(labels, mean_values, sem, xlabel, ylabel, name):
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, yerr=sem, color=['#2d7f5e', '#7f6d5f', '#3776ab', '#557f2d','silver'], align='center', ecolor='black', capsize=10)
	#plt.bar(labels, mean_values, yerr=sem, color=['#2d7f5e', '#7f6d5f', '#557f2d'], align='center', ecolor='black', capsize=10)

	plt.ylim(0.5, 1)
	#plt.ylim(1, 10**5)
	#plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	plt.xticks(rotation=45)
	plt.savefig(name+'.svg', format="svg")




if __name__ == '__main__':
	
	big_pool=load_data_functions.load_big_pool()

	nuclear=find_nuclear(big_pool)

	all_training=find_training_genes_functions.load_pos_neg_training()

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	hk, nuclear, golgi, mem, syngo, syndb, synsysnet=load_control_and_synapse_genes(big_pool, go_genes)

	syn=list(set(syngo)&set(syndb)&set(synsysnet))

	genelists=[syn, hk, nuclear, golgi, mem]
	genelist_names=['syn', 'hk', 'nuclear', 'golgi', 'mem']
	#genelist_diff_ci=compute_syn_control_ci(genelists, genelist_names, all_training)
	#print (genelist_diff_ci)

	pred_dfs, aucs=compute_pred_dfs_aucs(genelists, all_training)
	ebs=[]
	for item in pred_dfs:
		ci, errorbars=auc_bootstrap_errorbars(item)
		#print (errorbars)
		ebs.append(errorbars)

	labels=['Synapse', 'Housekeeping', 'Nuclear', 'Golgi App', 'Transmem']

	print (aucs)
	print (ebs)

	plot_controls_with_errorbar(labels, aucs, ebs, 'Gene Category', 'Predicted Recovery ROC', 'syn_control')

	genelist_diff_ci=compute_syn_control_ci(genelists, genelist_names, pred_dfs)
	print (genelist_diff_ci)

	ratios=calc_ctrl_tpr_fpr(syn, big_pool, all_training)
	print (ratios)
	
	


