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
sys.path.append('../read_data_functions/')
import graph_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores
import ROC_functions

if __name__ == '__main__':
	
	big_pool=find_training_genes_functions.load_big_pool()

	pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	syndb=load_data_functions.find_SynDB(big_pool)
	synsysnet=load_data_functions.find_synsysnet(big_pool)
	syn=list(set(syngo)&set(syndb)&set(synsysnet))

	db_list=[syngo, syndb, synsysnet, syn]
	db_labels=['syngo', 'syndb', 'synsysnet', 'syn']

	for i in range(len(db_list)):
		final, label, avg_score=ROC_functions.find_pred_labels_scores(db_list[i], all_training)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)
		print (auc)
		ROC_functions.save_roc_df(thresholds, tpr, fpr, db_labels[i])

	final, label, avg_score=ROC_functions.find_pred_labels_scores(syn, all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	

	graph_functions.plot_mean_ROC(tpr, fpr, auc, 'syn')
