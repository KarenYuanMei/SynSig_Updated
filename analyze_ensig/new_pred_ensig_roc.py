#Goal: 1) calculate the ROC of the predicted scores for predicting the synapse genes from SynGO, SynDB, and SynSysNet
#.     2) print the thresholds, tpr, fpr for each evaluation into csv

import pandas as pd
import numpy as np
import csv

import ddot
from ddot import Ontology

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

import sys
sys.path.append('../graph_functions/')
import graph_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores
import ROC_functions

sys.path.append('../run_ML/')
import predict_ensig_genes
import compare_controls

if __name__ == '__main__':
	
	nb_pool=predict_ensig_genes.find_nonbrain_common_pool()

	hk=compare_controls.find_hk(nb_pool)

	pos, neg, all_training=find_training_genes_functions.load_ensig_pos_neg_training()

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	syngo=load_data_functions.find_syngo(nb_pool, go_genes)
	syndb=load_data_functions.find_SynDB(nb_pool)
	synsysnet=load_data_functions.find_synsysnet(nb_pool)
	syn=list(set(syngo)&set(syndb)&set(synsysnet))

	# db_list=[syngo, syndb, synsysnet, syn]
	# db_labels=['syngo', 'syndb', 'synsysnet', 'syn']


	# for i in range(len(db_list)):
	# 	final, label, avg_score=ROC_functions.find_pred_labels_scores(db_list[i], all_training)
	# 	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)
	# 	print (auc)
	# 	ROC_functions.save_roc_df(thresholds, tpr, fpr, db_labels[i], 'nb')

	# final, label, avg_score=ROC_functions.find_pred_labels_scores(syn, all_training)
	# fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	

	# graph_functions.plot_single_ROC(tpr, fpr, auc, 'nb')

	ensig_hk_labels=['Non-Brain', 'Housekeeping']
	ensig_hk=[syngo, hk]

	for i in range(len(ensig_hk)):

		final, label, avg_score=ROC_functions.find_pred_labels_scores(ensig_hk[i], all_training)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	

		graph_functions.plot_single_ROC(tpr, fpr, auc, ensig_hk_labels[i])

	#graph_functions.plot_annotate_ROC_controls(tpr, fpr, auc)


