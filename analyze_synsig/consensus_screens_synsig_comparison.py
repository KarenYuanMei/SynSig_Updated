#Goal: 1) draw the ROC curve for predicted synapse scores with consensus mass spectrometry genes as metric
#      2) annotate how syngo and synsig compare
	#  3) Figure 3F



import pandas as pd
import numpy as np
import csv

import ddot
from ddot import Ontology

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

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
	#ref_list is consensus positive genes
	#genelist is prediction list, for example synsig
	#big_pool is the set of all possible genes
	#all_training is all genes used for training

	ref_list=list(set(ref_list)-set(all_training))
	genelist=list(set(genelist)-set(all_training))
	big_pool=list(set(big_pool)-set(all_training))

	tp=ref_list
	found_pos=list(set(tp)&set(genelist))
	tpr=float(len(found_pos)/len(tp))

	fp=list(set(genelist)-set(ref_list))
	all_negatives=list(set(big_pool)-set(ref_list))
	fpr=float(len(fp)/(len(all_negatives))) #divide the number of false positives by the total number of ground-truth negative genes
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

def plot_annotate_ROC_mass_spec(tpr, fpr, auc):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')

	plt.plot(fpr, tpr, color='maroon',
	         label=r'ROC (AUC = %0.2f)' % (auc),
	         lw=2, alpha=.8)

	#without training genes: {'synsig': (0.5786802030456852, 0.08594885829128539), 'syngo': (0.22081218274111675, 0.022230354038971733)}

	plt.plot(0.103, 0.579, color='purple', marker='o', markersize=10)
	plt.annotate('SynSig', color='purple', xy=(0.103, 0.579), xytext=(0.103+0.05, 0.579), arrowprops=dict(facecolor='purple', lw=2, arrowstyle='->'))

	plt.plot(0.025, 0.22, color='#7f6d5f', marker='o', markersize=10)
	plt.annotate('SynGO', color='#7f6d5f', xy=(0.025, 0.22), xytext=(0.025+0.05, 0.22), arrowprops=dict(facecolor='#7f6d5f', lw=2, arrowstyle='->'))

	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')


	plt.grid(False)
	# show the legend
	plt.legend()
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig('annotate_ROC_ms.svg', format="svg")


if __name__ == '__main__':
	
	big_pool=load_data_functions.load_big_pool()

	all_training=find_training_genes_functions.load_pos_neg_training()
	
	#human_ont=find_GO_scores.find_GO_ont()
	#go_genes=human_ont.genes

	#syngo=load_data_functions.find_syngo(big_pool, go_genes)
	syngo=load_data_functions.find_full_syngo()
	syngo=list(set(syngo)&set(big_pool))

	synsig=load_data_functions.load_synsig()

	synsysnet=load_data_functions.find_synsysnet(big_pool)
	#synsysnet=load_data_functions.find_full_synsysnet()

	synDB=load_data_functions.find_SynDB(big_pool)
	#synDB=load_data_functions.find_full_SynDB()

	ctx=load_data_functions.find_adult_cortex(big_pool)

	striatum=load_data_functions.find_adult_striatum(big_pool)

	fetal=load_data_functions.find_fetal(big_pool)

	ngn2=load_data_functions.find_ngn2(big_pool)

	consensus_ms=list(set(ctx)&set(striatum)&set(fetal)&set(ngn2))
	#df=load_data_functions.load_predicted_df()
	#print (df)
	pred_df=load_data_functions.load_predicted_synsig_df()

	final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, consensus_ms, all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)

	graph_functions.plot_single_ROC(tpr, fpr, auc, 'consensus_ms')
	print (auc)

	ratios=calc_ctrl_tpr_fpr(consensus_ms, [synsig, syngo], ['synsig', 'syngo'], big_pool, all_training)
	print (ratios)

	#adult_consensus=list(set(ctx)&set(striatum))
	#fetal_consensus=list(set(fetal)&set(ngn2))

	plot_annotate_ROC_mass_spec(tpr, fpr, auc)

	
