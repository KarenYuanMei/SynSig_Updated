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

import define_gene_objects
from load_data_functions import get_gene_names
import find_training_genes_scores_functions 

# #predict new genes=========================================================

def find_new_gene_objects(new_genes, feature_value_dict):
	new_genes=sorted(new_genes)
	print ('new genes', len(new_genes))
	new_gene_objects=define_gene_objects.create_new_gene_list(new_genes, False, feature_value_dict)
	print (len(new_gene_objects))
	print ('DONE')
	return new_gene_objects

def find_synapse_new_pairs(new_genes, feature_value_dict, all_training_objects, pos):
	new_gene_objects=find_new_gene_objects(new_genes, feature_value_dict)
	positive_training_objects=define_gene_objects.find_gene_objects(all_training_objects, pos)
	print ('positive_training_objects', len(positive_training_objects))
	synapse_new_pairs=product(positive_training_objects, new_gene_objects)
	return synapse_new_pairs

if __name__ == '__main__':

	big_pool=find_training_genes_functions.load_big_pool()

	go_mat_filename='../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'

	pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

	feature_value_dict = define_gene_objects.create_feature_value_dict(big_pool)

	all_training_objects=define_gene_objects.define_all_training_objects(all_training, go_mat_filename, feature_value_dict)

	training_pairs=combinations(all_training_objects,2)
	print ('DONE training pairs for final rf')

	new_genes=list(set(big_pool)-set(all_training))
	synapse_new_pairs=find_synapse_new_pairs(new_genes, feature_value_dict, all_training_objects, pos)

	feature_list=define_gene_objects.define_features()

	data_test, data_gene1, data_gene2=define_gene_objects.find_new_array(synapse_new_pairs, feature_list)
	print (data_test.shape)
	train_pair_objects, X_train, y_train=define_gene_objects.create_input_pair_objects(training_pairs)
	print (X_train.shape)

	define_gene_objects.run_new_rf(X_train, y_train, data_test, data_gene1, data_gene2)

	define_gene_objects.find_avg_scores(new_genes)
