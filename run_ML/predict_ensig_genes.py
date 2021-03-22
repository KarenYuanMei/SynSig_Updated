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

def define_nb_training_test_pairs(pos, neg, all_training, nb_pool, feature_list):
	training_gene_names, test_gene_names=find_training_genes_functions.define_crossvalidation_genes(pos, neg, 'nb')

	feature_value_dict = define_gene_objects.create_feature_value_dict(nb_pool, feature_list)

	go_mat_filename='../run_ML/ML_output/training_genes/nb_GO_training_score_matrix_for_big_pool_genes.csv'

	all_training_objects=define_gene_objects.define_all_training_objects(all_training, go_mat_filename, feature_value_dict, feature_list)

	training_pairs=combinations(all_training_objects,2)
	print ('DONE training pairs for final rf')

	new_genes=list(set(nb_pool)-set(all_training))
	print ('new genes', len(new_genes))
	synapse_new_pairs=find_synapse_new_pairs(new_genes, feature_value_dict, all_training_objects, pos)
	return training_pairs, synapse_new_pairs

nb_pool=find_nonbrain_common_pool()
syngo_file='../correct_db/corr_syngo_cc.csv'
syngo=load_data_functions.get_gene_names(syngo_file)

go_human=find_GO_scores.find_GO_ont()
GO_genes=go_human.genes

pos, neg, all_training=find_training_genes_functions.find_training_pos_neg(syngo, nb_pool, GO_genes)
nb_pos_df=find_training_genes_functions.make_genes_csv(pos, 'nb', 'positives')
nb_neg_df=find_training_genes_functions.make_genes_csv(neg, 'nb', 'negatives')

find_GO_scores.define_GO_score_matrix(pos, neg, go_human, 'nb')

feature_list=define_features.load_nonbrain_features()
training_pairs, synapse_new_pairs=define_nb_training_test_pairs(pos, neg, all_training, nb_pool, feature_list)

data_test, data_gene1, data_gene2=define_gene_objects.find_new_array(synapse_new_pairs, feature_list)
print (data_test.shape)
train_pair_objects, X_train, y_train=define_gene_objects.create_input_pair_objects(training_pairs)
print (X_train.shape)

df=define_gene_objects.run_new_rf(X_train, y_train, data_test, data_gene1,data_gene2, 100, 50, 2)
df.to_csv('../run_ML/ML_output/nb_new_all_gene_predictions.csv')
