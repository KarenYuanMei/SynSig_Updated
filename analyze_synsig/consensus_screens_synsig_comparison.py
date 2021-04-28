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

def calc_syn_tpr_fpr(ref_list, genelist, big_pool, all_training):
	ref_list=list(set(ref_list)-set(all_training))
	genelist=list(set(genelist)-set(all_training))
	tp=ref_list
	found_pos=list(set(tp)&set(genelist))
	tpr=float(len(found_pos)/len(tp))

	fp=list(set(genelist)-set(ref_list))
	tn=list(set(big_pool)-set(ref_list))
	fpr=float(len(fp)/(len(fp)+len(tn)))
	return tpr, fpr

def calc_ctrl_tpr_fpr(ref_list, genelists, genelist_names, big_pool, all_training):

	ratios={}
	#controls=[hk, golgi, mem]
	#control_names=['hk', 'golgi', 'mem']
	for i in range(len(genelists)):
		tpr, fpr=calc_syn_tpr_fpr(ref_list, genelists[i], big_pool, all_training)
		print (tpr, fpr)

		ratios[genelist_names[i]]=(tpr, fpr)
	return ratios


if __name__ == '__main__':
	
	big_pool=load_data_functions.load_big_pool()

	all_training=find_training_genes_functions.load_pos_neg_training()
	print (all_training[:5])

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	syngo=load_data_functions.find_syngo(big_pool, go_genes)

	synsig=load_data_functions.load_synsig()

	synsysnet=load_data_functions.find_synsysnet(big_pool)

	synDB=load_data_functions.find_SynDB(big_pool)

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

	graph_functions.plot_single_ROC(tpr, fpr, auc, 'consensus_ms')
	print (auc)

	ratios=calc_ctrl_tpr_fpr(consensus_ms, [synsig, syngo, synsysnet, synDB], ['synsig', 'syngo', 'synsysnet', 'synDB'], big_pool, all_training)
	print (ratios)

	adult_consensus=list(set(ctx)&set(striatum))
	fetal_consensus=list(set(fetal)&set(ngn2))

	final, label, avg_score=ROC_functions.find_pred_labels_scores(adult_consensus, all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)

	#graph_functions.plot_single_ROC(tpr, fpr, auc, 'consensus_ms')
	print (auc)

	ratios=calc_ctrl_tpr_fpr(adult_consensus, [synsig, syngo, synsysnet, synDB], ['synsig', 'syngo', 'synsysnet', 'synDB'], big_pool, all_training)
	print (ratios)

	final, label, avg_score=ROC_functions.find_pred_labels_scores(fetal_consensus, all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)

	#graph_functions.plot_single_ROC(tpr, fpr, auc, 'consensus_ms')
	print (auc)

	ratios=calc_ctrl_tpr_fpr(fetal_consensus, [synsig, syngo, synsysnet, synDB], ['synsig', 'syngo', 'synsysnet', 'synDB'], big_pool, all_training)
	print (ratios)
