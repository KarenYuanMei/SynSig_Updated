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


big_pool=find_training_genes_functions.load_big_pool()

pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

human_ont=find_GO_scores.find_GO_ont()
go_genes=human_ont.genes

hk=find_hk(big_pool)
golgi=find_golgi(big_pool)
mem=find_mem(big_pool)
syngo=find_syngo(big_pool, go_genes)

print (len(hk))
print (len(golgi))
print (len(mem))
print (len(syngo))

final, labels, avg_scores=ROC_functions.find_pred_labels_scores(hk, all_training)
fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(labels, avg_scores)
print (auc)