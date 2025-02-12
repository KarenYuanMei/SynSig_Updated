#Goal: 1) define training genes and GO scores
#	   2) five-fold cross-validation
#      3) compare regressors on 5-fold crossvalidation

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
import define_features
import time


#load all resource gene lists: syngo_cc genes, big_pool(all genes in the features matrix), GO_genes
def load_resource_gene_lists(syngo_file, index_file):
	syngo=load_data_functions.get_gene_names(syngo_file)

	big_pool=load_data_functions.get_gene_names(index_file)

	GO_human=find_GO_scores.find_GO_ont()
	GO_genes=GO_human.genes
	return syngo, big_pool, GO_genes

#for each fold in the crossvalidation, find the training genes, featuers, and GO_Scores; and the validation genes, features, and GO_scores
def find_crossvalidate_input(all_training_objects, feature_list, pos, i):

	training_gene_names, test_gene_names=find_training_genes_functions.load_crossvalidation_genes(i)

	#define each fold of training gene objects with features and scores
	training_gene_objects=define_gene_objects.find_gene_objects(all_training_objects, training_gene_names)
	print ('training_gene_objects', len(training_gene_objects))
	#construct training gene object pairs
	training_pairs=combinations(training_gene_objects,2)
	
	#find the positive training objects:
	positive_training_genes=find_training_genes_functions.find_pos_genes_in_training(training_gene_names, pos)
	positive_training_objects=define_gene_objects.find_gene_objects(all_training_objects, positive_training_genes)
	print ('positive_training_objects', len(positive_training_objects))

	#find the train_test_object_pairs:
	test_gene_objects=define_gene_objects.find_gene_objects(all_training_objects, test_gene_names)
	train_test_pairs=product(positive_training_objects, test_gene_objects)
	
	print ("DONE3")

	training_gene_pair_objects, training_feature_array, training_score=define_gene_objects.create_input_pair_objects(training_pairs, feature_list)

	print ('DONE')
	train_test_gene_pair_objects, tt_feature_array, tt_score=define_gene_objects.create_input_pair_objects(train_test_pairs, feature_list)
	print ('DONE')

	return training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score

#run adaboost and also time each fold of crossvalidation
def time_adaboost(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i):
	start = time.time()
	ada_df=regressor_functions.run_adaboost(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i)
	end = time.time()
	print('adaboost time', end - start)
	return ada_df

def time_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, kernel, i):
	start = time.time()
	svm_df=regressor_functions.run_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, kernel, i)
	end = time.time()
	print('svm time', end - start)
	return svm_df

def time_svm_poly(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, poly_number, i):
	start = time.time()
	svm_df=regressor_functions.run_svm_poly(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, poly_number, i)
	end = time.time()
	print('svm time', end - start)
	return svm_df

def time_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, feature_list, i):
	start = time.time()
	rf_df=regressor_functions.run_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, feature_list, i)
	end = time.time()
	print('random forest time', end - start)
	return rf_df

def compare_regressors(all_training_objects, pos, feature_list):

	for i in range(5):
		#define each fold of training and test genes:
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=find_crossvalidate_input(all_training_objects, pos, i)

		X_train, X_test, y_train, y_test=regressor_functions.redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)
		
		ada_df=time_adaboost(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, i)

		rf_df=time_random_forest(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, feature_list, i)

		poly3_df=time_svm_poly(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 3, i)

		poly4_df=time_svm_poly(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 4, i)

		rbf_df=time_svm_regressor(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 'rbf', i)

		sigmoid_df=time_svm_regressor(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 'sigmoid', i)

	return ada_df, rf_df, poly3_df, poly4_df, rbf_df, sigmoid_df


def run_cv_opt_rf(all_training_objects, pos, feature_list):
	
	for i in range(5):
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=find_crossvalidate_input(all_training_objects, pos, i)
		X_train, X_test, y_train, y_test=regressor_functions.redefine_input(training_feature_array, tt_feature_array, training_score, tt_score)
		regressor_functions.sweep_param_rf(training_gene_pair_objects, X_train, y_train, train_test_gene_pair_objects, X_test, y_test, 100, 50, 2)


if __name__ == '__main__':
	syngo_file='../../correct_db/corr_syngo_cc.csv'
	index_file='../../../SynSig/synsig_random_forest/big_pool_genes_index.csv'
	
	syngo, big_pool, GO_genes=load_resource_gene_lists(syngo_file, index_file)

	pos, neg, all_training=find_training_genes_functions.define_pos_neg_training(syngo, big_pool, GO_genes)

	go_mat_filename='../../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'

	feature_value_dict = define_gene_objects.create_feature_value_dict(big_pool)
	print ("DONE1")

	all_training_objects=define_gene_objects.define_all_training_objects(all_training, go_mat_filename, feature_value_dict)

	pos_chunks, neg_chunks=find_training_genes_functions.find_pos_neg_chunks(pos, neg)

	training_genes, test_genes=find_training_genes_functions.define_crossvalidation_genes(pos, pos_chunks, neg, neg_chunks)

	df=fivefold_crossvalidate_rf(pos, pos_chunks, neg, neg_chunks)
	#feature_list=define_features.load_all_features()
	#compare_regressors(all_training_objects, pos, feature_list)

	feature_list=define_features.load_filtered_features()
	run_cv_opt_rf(all_training_objects, pos, feature_list)

