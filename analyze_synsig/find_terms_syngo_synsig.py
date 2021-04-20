#Goal: find which SynGO terms were represented by training; how many were recovered by SynSig

import pandas as pd

import csv

import numpy as np

import ddot
from ddot import Ontology

import sys
sys.path.append('../ML_functions/')
import find_GO_scores
import find_training_genes_functions

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def term_index_to_go(idx_list, terms):
	go_index=[]
	for item in idx_list:
		go=terms[item]
		go_index.append(go)
	return go_index

def find_gene_to_terms(genelist, ont):
	terms=ont.terms
	gene_terms=[]
	for gene in genelist:
		idx_list=ont.gene_2_term[gene]
		go_index=term_index_to_go(idx_list, terms)
		#print (go_index)
		gene_terms.append(go_index)
	return gene_terms


def find_training_terms(ont):
	pos=find_training_genes_functions.load_pos_training()
	print (len(pos))

	training_terms=find_gene_to_terms(pos, ont)

	training_terms = [item for sublist in training_terms for item in sublist]
	training_terms=list(set(training_terms))

	print ('training', len(training_terms))
	return training_terms

def find_predicted_terms(ont):
	synsig=load_data_functions.load_synsig()
	print (len(synsig))

	overlap=list(set(synsig)&set(ont.genes))
	pred_terms=find_gene_to_terms(overlap, ont)

	pred_terms = [item for sublist in pred_terms for item in sublist]
	pred_terms=list(set(pred_terms))

	print ('pred', len(pred_terms))
	return pred_terms

def gene_to_names(genelist, ont):
	names=[]
	genes=ont.genes
	for idx in genelist:
		name=genes[idx]
		names.append(name)
	return names


ont = Ontology.from_table('../source_data_files/correct_db/SynGO_BP.txt')
ont = ont.propagate(direction='forward', gene_term=True, term_term=False)
print (ont)

synsig=load_data_functions.load_synsig()
print (len(synsig))

branches=ont.parent_2_child['synapse process']
print (branches)

term_genes={}
for item in branches:
	genes=ont.term_2_gene[item]
	gene_no=len(genes)
	names=gene_to_names(genes, ont)
	overlap=list(set(names)&set(synsig))
	print (item, len(overlap))
	term_genes[item]=(names)

print (term_genes['synaptic signaling'])

predicted=load_data_functions.load_predicted_df()
print (predicted)

pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()
#pred_dfs, aucs=compute_pred_dfs_aucs(genelists, all_training)

# syngo_genes=ont.genes
# syngo_terms=ont.terms

# #print (ont.term_2_gene)
# gene_no_list=[]
# for item in syngo_terms:
# 	genes=ont.term_2_gene[item]
# 	gene_no=len(genes)
# 	gene_no_list.append(gene_no)

# term_gene_no=list(zip(syngo_terms, gene_no_list))
# term_gene_dict=dict(term_gene_no)
# #
# from operator import itemgetter

# N = 50

# res = dict(sorted(term_gene_dict.items(), key = itemgetter(1), reverse = True)[:N])
# print (str(res))
#terms=ont.terms

# gene_terms=[]
# for term in ont.terms: 
# 	genes=ont.term_2_gene[term]
# 	if len(genes)>0:
# 		gene_terms.append(term)

# print ('terms with genes', len(gene_terms))


# training_terms=find_training_terms(ont)

# pred_terms=find_predicted_terms(ont)

# total_terms=list(set(training_terms+pred_terms))
# print (len(total_terms))
# print (pred_terms)

# overlap=list(set(total_terms)&set(gene_terms))
# print (len(overlap))

# not_covered=list(set(gene_terms)-set(total_terms))
# print (not_covered)


