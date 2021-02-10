#Goal: 1) predict new synapse genes

import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

from itertools import combinations, combinations_with_replacement
from itertools import product
from collections import defaultdict

from define_gene_objects import define_features, Gene, PairOfGenes, find_input_features, load_feature, create_feature_value_dict, get_feature_value, create_GO_score_dict, create_gene_list, find_pos_genes_in_training, find_gene_objects, find_feature_array, create_input_pair_objects, run_adaboost, run_svm_regressor, run_random_forest, find_new_genes, run_new_rf, find_new_array, find_avg_scores
from load_data_functions import get_gene_names
from find_training_genes_scores_functions import make_genes_csv, make_mat_csv, random_select, find_pos_neg_input, divide_5fold, find_pos_neg_chunks, define_training_test, find_GO_ont, find_GO_score_matrix, find_input_gene_GO_scores
from run_train_crossvalidate_pipeline import define_all_training_objects


# #predict new genes=========================================================
def load_pos_neg_training():
	pos=get_gene_names('updated_positives.csv')
	neg=get_gene_names('updated_negatives.csv')
	all_training=pos+neg
	return pos, neg, all_training

def load_big_pool():
	index_file='../../SynSig/synsig_random_forest/big_pool_genes_index.csv'
	big_pool=get_gene_names(index_file)
	return big_pool

def find_new_gene_objects(new_genes):
	new_genes=sorted(new_genes)
	print ('new genes', len(new_genes))
	new_gene_objects=create_new_gene_list(new_genes, False, feature_value_dict)
	print (len(new_gene_objects))
	print ('DONE')
	return new_gene_objects

def find_synapse_new_pairs(all_training, all_training_objects, pos):
	new_gene_objects=find_new_gene_objects(all_training)
	positive_training_objects=find_gene_objects(all_training_objects, pos)
	print ('positive_training_objects', len(positive_training_objects))
	synapse_new_pairs=product(positive_training_objects, new_gene_objects)
	return synapse_new_pairs

go_mat_filename='../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'

pos, neg, all_training=load_pos_neg_training()

all_training_objects=define_all_training_objects(big_pool, all_training, go_mat_filename)

training_pairs=combinations(all_training_objects,2)
print ('DONE training pairs for final rf')

new_genes=find_new_genes(all_training)
synapse_new_pairs=find_synapse_new_pairs(all_training, all_training_objects, pos)

feature_list=define_features()

data_test, data_gene1, data_gene2=find_new_array(synapse_new_pairs, feature_list)
print (data_test.shape)
train_pair_objects, X_train, y_train=create_input_pair_objects(training_pairs)
print (X_train.shape)

run_new_rf(X_train, y_train, data_test, data_gene1, data_gene2)

find_avg_scores(new_genes)

