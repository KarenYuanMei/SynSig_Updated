#Goal: to show how predictions compare to housekeeping and other gene lists

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

#find housekeeping:==============
def find_hk(big_pool):
	hk=pd.read_csv('../gene_lists/Human_Mouse_Common.csv', sep=';')
	#print (hk)
	hk=hk['Human'].tolist()
	hk=list(set(hk)&set(big_pool))
	return hk

#find golgi:===================
def find_golgi(big_pool):
	golgi=load_data_functions.get_gene_names('../gene_lists/golgi_genes.csv')
	golgi=list(set(golgi)&set(big_pool))
	return golgi

#find transmembrane:==============
def find_mem(big_pool):
	transm=pd.read_csv('../gene_lists/Uniprot_transmembrane.csv')
	transm=transm['Gene names'].tolist()
	transm=[str(x) for x in transm]
	mem=[]
	for item in transm:
		entry=item[:item.find(' ')]
		mem.append(entry)
	mem=list(set(mem)&set(big_pool))
	return mem

def find_syngo(big_pool, go_genes):
	syngo_file='../correct_db/corr_syngo_cc.csv'
	syngo=load_data_functions.get_gene_names(syngo_file)
	syngo=list(set(syngo)&set(big_pool)&set(go_genes))
	return syngo

def find_SynDB(big_pool):
	synDB_file='../correct_db/SynDB.csv'
	syndb=load_data_functions.get_gene_names(synDB_file)
	syndb=list(set(syndb)&set(big_pool))
	return syndb

def find_synsysnet(big_pool):
	synsysnet_file='../correct_db/synsysnet.csv'
	synsysnet=load_data_functions.get_gene_names(synsysnet_file)
	synsysnet=list(set(synsysnet)&set(big_pool))
	return synsysnet


def compare_auc_bootstrap(set1_predictions,set2_predictions):
	#set1_predictions and set2_predictions should be the output from find_true_y
	#returns a confidence interval for the difference between the auc scores for the two sets
	scores = set1_predictions['avg_scores']

	set1_labels = set1_predictions['label']
	set2_labels = set2_predictions['label']

	num_bootstrap_samples = 1000

	bootstrapped_auc_diffs = []
	for i in range(num_bootstrap_samples):
		#indices = random.randint(0,len(scores))
		indices=list(np.random.randint(low = 0,high=len(scores),size=len(scores)))
		#print (indices)

		set1_auc = roc_auc_score(set1_labels[indices],scores[indices])
		set2_auc = roc_auc_score(set2_labels[indices],scores[indices])

		diff = set1_auc - set2_auc

		bootstrapped_auc_diffs.append(diff) 


	conf_interval_sizes = [0.95,0.99]
	conf_intervals = {}

	bootstrapped_auc_diffs.sort()

	for interval_size in conf_interval_sizes:
		print (interval_size)
		lower_bound_index = int(num_bootstrap_samples*((1-interval_size)/2))
		print (lower_bound_index)

		lower_bound = bootstrapped_auc_diffs[lower_bound_index]

		upper_bound_index = int(num_bootstrap_samples*(interval_size+((1-interval_size)/2)))
		print (upper_bound_index)
		upper_bound = bootstrapped_auc_diffs[upper_bound_index]

		conf_intervals[interval_size] = (lower_bound,upper_bound)

	return conf_intervals


big_pool=find_training_genes_functions.load_big_pool()

pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

human_ont=find_GO_scores.find_GO_ont()
go_genes=human_ont.genes

hk=find_hk(big_pool)
golgi=find_golgi(big_pool)
mem=find_mem(big_pool)
syngo=find_syngo(big_pool, go_genes)
syndb=find_SynDB(big_pool)
synsysnet=find_synsysnet(big_pool)

syn=list(set(syngo)&set(syndb)&set(synsysnet))

print (len(hk))
print (len(golgi))
print (len(mem))
print (len(syngo))

genelists=[syngo, hk, golgi, mem]

for item in genelists:
	final, labels, avg_scores=ROC_functions.find_pred_labels_scores(item, all_training)
	#print (final)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(labels, avg_scores)
	print (auc)


syn_final, labels, avg_scores=ROC_functions.find_pred_labels_scores(syn, all_training)
hk_final, labels, avg_scores=ROC_functions.find_pred_labels_scores(hk, all_training)
conf_interval=compare_auc_bootstrap(syn_final, hk_final)
print (conf_interval)
