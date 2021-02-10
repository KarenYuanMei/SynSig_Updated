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

from define_gene_objects import define_features, Gene, PairOfGenes, find_input_features, load_feature, create_feature_value_dict, get_feature_value, create_GO_score_dict, create_gene_list, find_pos_genes_in_training, find_gene_objects, find_feature_array, create_input_pair_objects, run_adaboost, run_svm_regressor, run_random_forest, find_data_genes, create_data_gene_list, run_new_rf, find_data_array, find_avg_scores
from load_data_functions import get_gene_names
from find_training_genes_scores_functions import make_genes_csv, make_mat_csv, random_select, find_pos_neg_input, divide_5fold, find_pos_neg_chunks, define_training_test, find_GO_ont, find_GO_score_matrix, find_input_gene_GO_scores

import time


def define_GO_score_matrix(pos, neg, GO_human):
	GO_score_matrix=find_input_gene_GO_scores(pos, neg, GO_human)
	make_mat_csv(GO_score_matrix, 'syngo_GO_training_score_matrix_for_big_pool_genes.csv')
	return GO_score_matrix

def load_GO_score_matrix(go_mat_filename):
	df=pd.read_csv(go_mat_filename, index_col=[0])
	return df

def find_training_pos_neg(syngo, big_pool, GO_genes):
	pos, neg=find_pos_neg_input(syngo, big_pool, GO_genes)
	print ('pos', len(pos), 'neg', len(neg))
	all_training=list(set(pos+neg))
	all_training=sorted(all_training)
	print(len(all_training))
	return pos, neg, all_training

#load all resource gene lists:
def load_resource_gene_lists(syngo_file, index_file):
	syngo=get_gene_names(syngo_file)

	big_pool=get_gene_names(index_file)

	GO_human=find_GO_ont()
	GO_genes=GO_human.genes
	return syngo, big_pool, GO_genes

#find the pos and neg training genes:
def define_pos_neg_training(syngo, big_pool, GO_genes):
	pos, neg, all_training=find_training_pos_neg(syngo, big_pool, GO_genes)
	pos_df=make_genes_csv(pos, 'updated', 'positives')
	neg_df=make_genes_csv(neg, 'updated', 'negatives')
	return pos, neg, all_training

#define all gene objects with features and GO scores:
def define_all_training_objects(big_pool, all_training, go_mat_filename):
	feature_value_dict = create_feature_value_dict(big_pool)
	print ("DONE1")
	#go_mat_filename='../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'
	go_score_mat=load_GO_score_matrix(go_mat_filename)
	all_training_objects = create_gene_list(all_training,False,feature_value_dict, go_score_mat)
	print (len(all_training_objects))
	print ("DONE2")
	return all_training_objects


def find_crossvalidate_input(pos, pos_chunks, neg, neg_chunks, i):
	training_gene_names, test_gene_names=define_training_test(pos, pos_chunks, neg, neg_chunks, i)
	training_df=make_genes_csv(training_gene_names, 'updated', 'training_genes_%s'%i)
	test_df=make_genes_csv(test_gene_names, 'updated', 'test_genes_%s'%i)

	#define each fold of training gene objects with features and scores
	training_gene_objects=find_gene_objects(all_training_objects, training_gene_names)
	print ('training_gene_objects', len(training_gene_objects))
	#construct training gene object pairs
	training_pairs=combinations(training_gene_objects,2)
	
	#find the positive training objects:
	positive_training_genes=find_pos_genes_in_training(training_gene_names, pos)
	positive_training_objects=find_gene_objects(all_training_objects, positive_training_genes)
	print ('positive_training_objects', len(positive_training_objects))

	#find the train_test_object_pairs:
	test_gene_objects=find_gene_objects(all_training_objects, test_gene_names)
	train_test_pairs=product(positive_training_objects, test_gene_objects)
	
	print ("DONE3")

	training_gene_pair_objects, training_feature_array, training_score=create_input_pair_objects(training_pairs)

	print ('DONE')
	train_test_gene_pair_objects, tt_feature_array, tt_score=create_input_pair_objects(train_test_pairs)
	print ('DONE')

	return training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score

def fivefold_crossvalidate_rf(pos, pos_chunks, neg, neg_chunks):
	for i in range(5):
		#define each fold of training and test genes:
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=find_crossvalidate_input(pos, pos_chunks, neg, neg_chunks, i)
		df=run_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i)
	return df

def time_adaboost(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i):
	start = time.time()
	ada_df=run_adaboost(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i)
	end = time.time()
	print('adaboost time', end - start)
	return ada_df

def time_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, kernel, i):
	start = time.time()
	svm_df=run_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, kernel, i)
	end = time.time()
	print('svm time', end - start)
	return svm_df

def time_svm_poly(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, poly_number, i):
	start = time.time()
	svm_df=run_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, poly_number, i)
	end = time.time()
	print('svm time', end - start)
	return svm_df

def time_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i):
	start = time.time()
	rf_df=run_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i)
	end = time.time()
	print('random forest time', end - start)
	return rf_df

def compare_regressors(pos, pos_chunks, neg, neg_chunks):
	for i in range(5):
		#define each fold of training and test genes:
		training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score=find_crossvalidate_input(pos, pos_chunks, neg, neg_chunks, i)
		
		ada_df=time_adaboost(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i)

		rf_df=time_random_forest(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, i)

		poly3_df=time_svm_poly((training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, '3', i))

		poly4_df=time_svm_poly((training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, '4', i))

		rbf_df=time_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, 'rbf', i)

		sigmoid_df=time_svm_regressor(training_gene_pair_objects, training_feature_array, training_score, train_test_gene_pair_objects, tt_feature_array, tt_score, 'sigmoid', i)

	return ada_df, rf_df, poly3_df, poly4_df, svm_df, sigmoid_df


if __name__ == '__main__':
	syngo_file='../correct_db/corr_syngo_cc.csv'
	index_file='../../SynSig/synsig_random_forest/big_pool_genes_index.csv'
	
	syngo, big_pool, GO_genes=load_resource_gene_lists(syngo_file, index_file)

	pos, neg, all_training=define_pos_neg_training(syngo, big_pool, GO_genes)

	go_mat_filename='../syngo_training/syngo_GO_training_score_matrix_for_big_pool_genes.csv'

	all_training_objects=define_all_training_objects(big_pool, all_training, go_mat_filename)

	pos_chunks, neg_chunks=find_pos_neg_chunks(pos, neg)

	#df=fivefold_crossvalidate_rf(pos, pos_chunks, neg, neg_chunks)
	compare_regressors(pos, pos_chunks, neg, neg_chunks)




