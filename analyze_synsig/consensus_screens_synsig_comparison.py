#Goal: 1) draw the ROC curve for predicted synapse scores with consensus mass spectrometry genes as metric
#      2) annotate how syngo and synsig compare



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

if __name__ == '__main__':
	
	big_pool=load_data_functions.load_big_pool()

	all_training=find_training_genes_functions.load_pos_neg_training()

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	syngo=load_data_functions.find_syngo(big_pool, go_genes)

	ctx=load_data_functions.find_adult_cortex(big_pool)

	striatum=load_data_functions.find_adult_striatum(big_pool)

	fetal=load_data_functions.find_fetal(big_pool)

	ngn2=load_data_functions.find_ngn2(big_pool)

	consensus_ms=list(set(ctx)&set(striatum)&set(fetal)&set(ngn2))
	print (consensus_ms[:5])

	#df=load_data_functions.load_predicted_df()
	#print (df)

	final, label, avg_score=ROC_functions.find_pred_labels_scores(consensus_ms, all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)
	print (auc)