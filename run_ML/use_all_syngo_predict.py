#Goal: 1) Final model: use all of SynGO as training genes

#note: run the file on cluster using run_final_model.sh file

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

from run_train_crossvalidate_pipeline import load_resource_gene_lists
from predict_new_synapse import find_synapse_new_pairs

#find negative training genes (same size as the positive training genes):

def find_syngo_negative_training(big_pool, GO_genes, syngo):
	big_pool_in_GO=list(set(big_pool)&set(GO_genes))
	negatives=list(set(big_pool_in_GO)-set(syngo))
	negatives.sort()
	print (negatives[:5])
	random.seed(1)
	pos_training=list(set(big_pool)&set(GO_genes)&set(syngo))
	sel=random.sample(negatives, len(pos_training))
	return sel

if __name__ == '__main__':
	

	#define positive syngo training genes:====================================

	#load syngo genes, all the genes in the features (big_pool), GO_genes
	syngo_file='../../correct_db/corr_syngo_cc.csv'
	index_file='../../../SynSig/synsig_random_forest/big_pool_genes_index.csv'
	syngo, big_pool, GO_genes=load_resource_gene_lists(syngo_file, index_file)

	#find intersection of syngo genes, all genes in the features (big_pool), and GO_genes:
	pos_training=list(set(syngo)&set(big_pool)&set(GO_genes))
	print (len(pos_training))

	#817 genes as positive training genes

	#define negative training genes:==================================================
	#find negative training genes: genes that are in big_pool and GO, but not in SynGO
	neg_training=find_syngo_negative_training(big_pool, GO_genes, syngo)
	print (len(neg_training))

	#compile all training genes:=========
	all_training=list(pos_training+neg_training)

	#find the new genes that need prediction scores:=================
	new_genes=list(set(big_pool)-set(all_training))
	print ('new genes', len(new_genes))
	print (new_genes[:5])


	#double check that the new genes do not overlap with new genes; len of overlap should be 0:
	overlap=list(set(pos_training)&set(new_genes))
	print (len(overlap))


	#find the GO score matrix for the positive and negative genes:=========================
	go_human=find_GO_scores.find_GO_ont()
	go_score_mat=find_GO_scores.define_GO_score_matrix(pos_training, neg_training, go_human, 'use_all_syngo')


	#find features==========================
	feature_list=define_features.load_filtered_features()

	feature_value_dict = define_gene_objects.create_feature_value_dict(big_pool, feature_list)


	#find pairs of training objects: pos-pos, pos-neg, neg-neg ===============================
	all_training_objects = define_gene_objects.create_gene_list(all_training,False,feature_value_dict, feature_list, go_score_mat)

	training_pairs=combinations(all_training_objects,2)
	print ('DONE training pairs for final rf')


	#construct pairs of new synapse genes with features========================================
	synapse_new_pairs=find_synapse_new_pairs(new_genes, feature_value_dict, all_training_objects, pos_training, feature_list)

	data_test, data_gene1, data_gene2=define_gene_objects.find_new_array(synapse_new_pairs, feature_list)
	print (data_test.shape)
	train_pair_objects, X_train, y_train=define_gene_objects.create_input_pair_objects(training_pairs, feature_list)
	print (X_train.shape)

	# #run the random forest model:====================================================

	forest, df=regressor_functions.run_new_rf(X_train, y_train, data_test, data_gene1,data_gene2, 100, 50, 2)
	pred_filename='../run_ML/ML_output/use_all_syngo_new_gene_predictions.csv'
	df.to_csv(pred_filename)

	#find the average semantic similarity score for each gene and output to '%s_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv':=======================
	regressor_functions.find_avg_scores(pred_filename, new_genes, 'all_syngo')
