#Goal: 1) evaluate how well the synsig model performs after removing different features

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
sys.path.append('../../read_data_functions/')
import load_data_functions

sys.path.append('../../ML_functions/')
import find_training_genes_functions 
import define_gene_objects
import regressor_functions
import define_features
import find_GO_scores
import ROC_functions

if __name__ == '__main__':
	feature_list=define_features.load_filtered_features()
	features_list.remove('mentha_kernel')

	training_pairs, synapse_new_pairs=define_training_test_pair_objects(feature_list)

	data_test, data_gene1, data_gene2=define_gene_objects.find_new_array(synapse_new_pairs, feature_list)
	print (data_test.shape)
	train_pair_objects, X_train, y_train=define_gene_objects.create_input_pair_objects(training_pairs, feature_list)
	print (X_train.shape)

	forest, df=regressor_functions.run_new_rf(X_train, y_train, data_test, data_gene1,data_gene2, 100, 50, 2)

	df.to_csv('remove_mentha_gene_predictions.csv')