#Goal: optimize random forest hyperparameters


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
import define_gene_objects
import regressor_functions
import find_training_genes_functions
import find_GO_scores

import time

import define_features


def sweep_parameters(all_training_objects, pos, tree_no, depth, split, name):	
	for i in range(5):
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=run_train_crossvalidate_pipeline.find_crossvalidate_input(all_training_objects, pos, i)
		X_train, X_test, y_train, y_test=regressor_functions.redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)
		df=regressor_functions.run_new_rf(X_train, y_train, new_test, new_gene1, new_gene2, tree_no, depth, split)
		df.to_csv('sweep_rf_%s_%s.csv'%s)


big_pool=find_training_genes_functions.load_big_pool()

pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

feature_list=define_features.load_all_features()
print (len(feature_list))

feature_value_dict = define_gene_objects.create_feature_value_dict(big_pool, feature_list)

print ("DONE1")

go_mat_filename='../../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'

all_training_objects=define_gene_objects.define_all_training_objects(all_training, go_mat_filename, feature_value_dict)

tree_no=[100, 125, 150, 175, 200, 225, 250, 275, 300]
for item in tree_no:
	sweep_parameters(all_training_objects, pos, item, None, 2, 'treeno')




