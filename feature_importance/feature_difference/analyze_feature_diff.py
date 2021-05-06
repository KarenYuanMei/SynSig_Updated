#Goal: analyze number of transcripts in the human genes as reported by Ensembl to determine if there are differences between synapse and non-synapse genes in this feature

#source: Ensembl

import csv
import numpy as np
import math
import pandas as pd
from scipy import stats

from mlxtend.evaluate import permutation_test

import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

from scipy import stats

from numpy.random import seed 
from numpy.random import randn 
from scipy.stats import mannwhitneyu 

import ddot
from ddot import Ontology
import random

import sys
sys.path.append('../../graph_functions/')
import graph_functions

sys.path.append('../../read_data_functions/')
import load_data_functions

sys.path.append('../../ML_functions/')
import find_training_genes_functions

plt.style.use('seaborn-deep')
matplotlib.rcParams.update({'font.size': 22})

def find_permutation(positives, negatives):
	p_value=permutation_test(positives, negatives, method='approximate', num_rounds=10000, seed=0)
	#print (p_value)
	return p_value

def find_students_test(positives, negatives):
	tstat, pvalue=stats.ttest_ind(positives, negatives, equal_var = False)
	return tstat, pvalue


#find genes that are not predicted to be in SynSig, but are in the big_pool
def find_synapse_negative_pool(synsig):
	big_pool=load_data_functions.load_big_pool()
	all_training=find_training_genes_functions.load_pos_neg_training()
	no_train_pool=list(set(big_pool)-set(all_training))
	negative_pool=list(set(no_train_pool)-set(synsig))
	return negative_pool

def find_gene_feature_values(df, genelist):
	df=df.set_index('Norm_Symbol')
	focused_df=df.loc[genelist]
	print (focused_df)
	focused_values=focused_df.iloc[:,-1].tolist()
	#print (focused_values)
	#print (focused_values[:5])
	return focused_values


if __name__ == '__main__':
	
	features=['ENSEMBL_aa_length', "cds_length", "exon_no", "gc_content", "trans_len", "gene_length", "Ensembl_isoform_no", "trans_count", "pFAM_domain_number", "Phosphosite_hu_no", "protein_mass"]

	for item in features:
		filename='../../../../SynSig/features/normalized_%s.csv'%item

		data_type=item
		#print (item)

		df=pd.read_csv(filename)
		#print (df)

		synsig_pos=load_data_functions.load_synsig()
		#print (len(synsig))

		synsig_neg=find_synapse_negative_pool(synsig_pos)

		#print (len(synsig_neg))

		syn_val=find_gene_feature_values(df, synsig_pos)
		#print (syn_val[:5])
		neg_val=find_gene_feature_values(df, synsig_neg)
		#print (neg_val[:5])

		p_value=find_permutation(syn_val, neg_val)
		print (p_value)

# 	tstat, pvalue=find_students_test(syn_val, neg_val)
# 	print (tstat, pvalue)

