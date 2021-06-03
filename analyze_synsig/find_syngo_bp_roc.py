#Goal: find the ROC recovery of syngo BP functions by synsig

import pandas as pd

import csv

import numpy as np

import ddot
from ddot import Ontology

import sys
sys.path.append('../ML_functions/')
import find_GO_scores
import find_training_genes_functions
import ROC_functions

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

def find_branch_genes_dict(ont):
	term_genes={}
	for item in branches:
		genes=ont.term_2_gene[item]
		gene_no=len(genes)
		names=gene_to_names(genes, ont)
		term_genes[item]=(names)
	return term_genes

def plot_single_ROC(tpr, fpr, auc, name):

	plt.plot(fpr, tpr,
	         label='%s'%name+r' ROC (AUC = %0.2f)' % (auc),
	         lw=2, alpha=.8)

	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')
	plt.grid(False)
	# show the legend
	plt.legend()
	plt.xlim([0, 1])
	plt.ylim([0, 1])
		# show the plot
	#plt.show()
	#plt.savefig('%s_ROC.svg'%name, format="svg")

def load_syngo_bp_ont():
	ont = Ontology.from_table('../source_data_files/correct_db/SynGO_BP.txt')
	ont = ont.propagate(direction='forward', gene_term=True, term_term=False)
	return ont

def find_syngo_bp_major_branches(ont):
	branches=ont.parent_2_child['synapse process']
	branches=branches[2:-1]
	return branches

def find_branch_roc(branches, term_genes, all_genes, all_training, predicted_df):
	branch_roc={}
	for item in branches:
		genelist=term_genes[item]
		genes_remove=list(set(all_genes)-set(genelist))
		print (len(genes_remove))
		genes_remove=list(set(genes_remove+all_training))

		final_df, labels, avg_scores=ROC_functions.find_pred_labels_scores(predicted_df, genelist, genes_remove)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(labels, avg_scores)
		print (auc)
		branch_roc[item]=(fpr, tpr, auc)
	return branch_roc

def plot_syngo_bp_roc(branches, branch_roc):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')
	for item in branches:
		tpr, fpr, auc=branch_roc[item]
		plot_single_ROC(fpr, tpr, auc, item)
		plt.savefig('syngo_bp_ROC.svg', format='svg')

if __name__ == '__main__':
	
	#load the syngo BP ontology
	ont=load_syngo_bp_ont()

	#find the children directly beneath the root of syngo BP; remove presynaptic and postsynaptic process (redundant) and remove pathway (way too small and no longer on syngoportal)
	branches=find_syngo_bp_major_branches(ont)

	#load synsig genes
	synsig=load_data_functions.load_synsig()

	#find major branch with respective gene name dictionary:
	term_genes=find_branch_genes_dict(ont)

	#find all genes in the major branches:
	all_genes=term_genes.values()
	all_genes = [item for sublist in all_genes for item in sublist]

	#load the predicted score df
	predicted=load_data_functions.load_predicted_synsig_df()

	#find all training genes from synsig predictions:
	all_training=find_training_genes_functions.load_pos_neg_training()

	#find the recovery roc for each syngo BP major branch:
	branch_roc=find_branch_roc(branches, term_genes, all_genes, all_training, predicted)

	#plot the ROCs:
	plot_syngo_bp_roc(branches, branch_roc)


