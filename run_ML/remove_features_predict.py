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
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions 
import define_gene_objects
import regressor_functions
import define_features
import find_GO_scores
import ROC_functions

import predict_new_synapse

def find_predicted_score_df(feature_list):
	training_pairs, synapse_new_pairs=predict_new_synapse.define_training_test_pair_objects(feature_list)

	data_test, data_gene1, data_gene2=define_gene_objects.find_new_array(synapse_new_pairs, feature_list)
	print (data_test.shape)
	train_pair_objects, X_train, y_train=define_gene_objects.create_input_pair_objects(training_pairs, feature_list)
	print (X_train.shape)

	forest, df=regressor_functions.run_new_rf(X_train, y_train, data_test, data_gene1,data_gene2, 100, 50, 2)

	

	return df

def find_avg_score_df(filename, name):
	big_pool=load_data_functions.load_big_pool()

	all_training=find_training_genes_functions.load_pos_neg_training()

	new_genes=list(set(big_pool)-set(all_training))
	
	avg_scores_df=regressor_functions.find_avg_scores(filename, new_genes, name)
	return avg_scores_df

def find_ref_lists():
	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	big_pool=load_data_functions.load_big_pool()

	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	syndb=load_data_functions.find_SynDB(big_pool)
	synsysnet=load_data_functions.find_synsysnet(big_pool)
	syn=list(set(syngo)&set(syndb)&set(synsysnet))
	return syngo, syn

if __name__ == '__main__':
	syngo, syn=find_ref_lists()
	all_training=find_training_genes_functions.load_pos_neg_training()

	# load the feature list, remove the feature of interest:
	feature_list=define_features.load_filtered_features()
	features_to_del=['gene_length']
	for item in features_to_del:
		feature_list.remove(item)
		print (feature_list)

		pred_scores_df=find_predicted_score_df(feature_list)
		filename='remove_%s_gene_predictions.csv'%item
		pred_scores_df.to_csv(filename)

	# find the average scores for each new gene:
		avg_scores_df=find_avg_score_df(filename, item)

		#evaluate the predicted scores with ROC

		#pred_filename='%s_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv'%item

		#avg_scores_df=pd.read_csv(pred_filename)

		final, label, avg_score=ROC_functions.find_pred_labels_scores(avg_scores_df, syn, all_training)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	
		print (auc)

		#graph_functions.plot_single_ROC(tpr, fpr, auc, 'syn_remove_ppi')

		final, label, avg_score=ROC_functions.find_pred_labels_scores(avg_scores_df, syngo, all_training)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	
		print (auc)

		#graph_functions.plot_single_ROC(tpr, fpr, auc, 'syngo_remove_ppi')
