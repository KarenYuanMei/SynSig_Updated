#Goal: 1) predict ensig genes

import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

from itertools import combinations, combinations_with_replacement
from itertools import product
from collections import defaultdict

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions 
import define_gene_objects
import define_features
import find_GO_scores


def find_feature_genes(filename):
	string_files=['pFAM_domain', 'mentha_source_feature','biogrid_source_feature', 'bioplex_source_feature', 'chr_no_source_feature']
	kernel_file=['mentha_kernel']
	gtex_rna_file=['gtex_rna_kernel']
	bioplex_file=['bioplex_kernel']
	if filename in string_files:
		df = pd.read_csv('../../../SynSig/features/normalized_%s.csv'%filename,converters={"Interactors": lambda x: x.strip("[]").split(", ")})
		symbol=df['Norm_Symbol']
		df.drop(labels=['Norm_Symbol', 'Genes'], axis=1,inplace = True)
		df.insert(0, 'Genes', symbol)
		df=df.set_index('Genes')
		idx=list(df.index)

	elif filename in kernel_file:
		df=pd.read_csv('../../../../Network_propagation/propagate_synapse/mentha_kernel.csv', index_col=[0])
		idx=list(df.index)

	elif filename in bioplex_file:
		df=pd.read_csv('../network_analysis/run_rwalk/bioplex_kernel.csv', index_col=[0])
		idx=list(df.index)

	else:
		df=pd.read_csv('../../../SynSig/features/normalized_%s.csv'%filename)

		symbol=df['Norm_Symbol']
		df.drop(labels=['Norm_Symbol', 'Genes'], axis=1,inplace = True)
		df.insert(0, 'Genes', symbol)
		#print (df)
		df=df.set_index('Genes')
		idx=list(df.index)
		
	return idx

def find_nonbrain_common_pool():
	nonbrain_features=define_features.load_nonbrain_features()

	idx_list=[]
	for feature in nonbrain_features:
		idx=find_feature_genes(feature)
		#print (idx[:5])
		idx_list.append(idx)

	common=set.intersection(*[set(list) for list in idx_list])
	common=list(set(common))
	#print (len(common))
	return common

nb_pool=find_nonbrain_common_pool()
syngo_file='../correct_db/corr_syngo_cc.csv'
syngo=load_data_functions.get_gene_names(syngo_file)

go_human=find_GO_scores.find_GO_ont()
GO_genes=go_human.genes

pos, neg, all_training=find_training_genes_functions.find_training_pos_neg(syngo, nb_pool, GO_genes)
nb_pos_df=find_training_genes_functions.make_genes_csv(pos, 'nb', 'positives')
nb_neg_df=find_training_genes_functions.make_genes_csv(neg, 'nb', 'negatives')

training_gene_names, test_gene_names=find_training_genes_functions.define_crossvalidation_genes(pos, neg, 'nb')
