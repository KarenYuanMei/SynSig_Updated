#Goal: optimize random forest hyperparameters


import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

from itertools import combinations, combinations_with_replacement
from itertools import product

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import define_gene_objects
import regressor_functions
import find_training_genes_functions
import find_GO_scores
import run_train_crossvalidate_pipeline

import time

import define_features


def sweep_parameters(all_training_objects, feature_list, pos, tree_no, depth, split, name, param):	
	for i in range(5):
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=run_train_crossvalidate_pipeline.find_crossvalidate_input(all_training_objects, feature_list, pos, i)
		X_train, X_test, y_train, y_test=regressor_functions.redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)
		
		sweep_tree_no(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, i)
		sweep_max_depth(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, i)
		sweep_split(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, i)


def sweep_tree_no(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test):
	tree_no=[100, 125, 150, 175, 200, 225, 250, 275, 300]
	for item in tree_no:
		df=regressor_functions.sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, item, None, 2)
		df.to_csv('sweep_rf_%s_%s_%s.csv'%('tree', item, i))

def sweep_max_depth(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, i):
	max_depth=[10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
	for item in max_depth:
		df=regressor_functions.sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 100, item, 2)
		df.to_csv('sweep_rf_%s_%s_%s.csv'%('max_depth', item, i))

def sweep_split(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, i):
	min_samples_split=[2, 3, 4, 5, 10]
	for item in min_samples_split:
		df=regressor_functions.sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 100, 50, item)
		df.to_csv('sweep_rf_%s_%s_%s.csv'%('split', item, i))

if __name__ == '__main__':
	
	big_pool=find_training_genes_functions.load_big_pool()

	pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

	feature_list=define_features.load_filtered_features()
	print (len(feature_list))

	feature_value_dict = define_gene_objects.create_feature_value_dict(big_pool, feature_list)

	go_mat_filename='../../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'

	all_training_objects=define_gene_objects.define_all_training_objects(all_training, go_mat_filename, feature_value_dict, feature_list)

	for i in range(5):
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=run_train_crossvalidate_pipeline.find_crossvalidate_input(all_training_objects, feature_list, pos, i)
		X_train, X_test, y_train, y_test=regressor_functions.redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)

		max_depth=[40, 50, 60, 70, 80, 90, 100]
		for item in max_depth:
			df=regressor_functions.sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 100, item, 2)
			df.to_csv('../run_ML/ML_output/sweep_param_results/sweep_rf_%s_%s_%s.csv'%('max_depth', item, i))

		min_samples_split=[2, 3, 4, 5, 10]
		for item in min_samples_split:
			df=regressor_functions.sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 100, 50, item)
			df.to_csv('../run_ML/ML_output/sweep_param_results/sweep_rf_%s_%s_%s.csv'%('split', item, i))




