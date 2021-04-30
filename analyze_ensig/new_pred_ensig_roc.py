#Goal: 1) calculate the ROC of the predicted scores for predicting the synapse genes from SynGO, SynDB, and SynSysNet
#.     2) print the thresholds, tpr, fpr for each evaluation into csv

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

sys.path.append('../run_ML/')
import predict_ensig_genes
import compare_controls

def find_nb_all_syngo(go_genes):
	nb_pool=predict_ensig_genes.find_nonbrain_common_pool()
	syngo_file='../source_data_files/correct_db/corr_syngo_cc.csv'
	syngo=load_data_functions.get_gene_names(syngo_file)
	syngo_nb=list(set(nb_pool)&set(syngo)&set(go_genes))
	print (len(syngo_nb))

	big_pool_filename='../../../SynSig/synsig_random_forest/big_pool_genes_index.csv'
	big_pool=load_data_functions.get_gene_names(big_pool_filename)
	syngo_big_pool=list(set(big_pool)&set(syngo)&set(go_genes))
	print (len(syngo_big_pool))

	return syngo_nb, syngo_big_pool

def plot_syngo_bargraph(labels, mean_values, xlabel, ylabel, name):
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, align='center', color='#2d7f5e', ecolor='black', capsize=10)

	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	#plt.xticks(rotation=45)
	plt.savefig(name+'.svg', format="svg")
	plt.close()

if __name__ == '__main__':
	
	#find the pool of non-brain genes
	nb_pool=predict_ensig_genes.find_nonbrain_common_pool()

	#find housekeeping genes in the pool of non-brain genes
	hk_nb=compare_controls.find_hk(nb_pool)

	#find the positive, negative, and all training genes used for predicting ensig:
	pos, neg, ensig_training=find_training_genes_functions.load_ensig_pos_neg_training()


    #find SynGo genes in the non-brain pool (syngo_nb)
    #find SynGO genes in big pool (syngo_big_pool)
	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes
	syngo_nb, syngo_big_pool=find_nb_all_syngo(go_genes)


	labels=['All_SynGO', 'Non-Brain SynGO']
	graph_functions.plot_venn2(syngo_big_pool, syngo_nb, labels, 'nb_vs_all_syngo')

	gene_no=[len(syngo_big_pool), len(syngo_nb)]
	xlabel='Gene Categories'
	ylabel='Gene No'
	plot_syngo_bargraph(labels, gene_no, xlabel, ylabel, 'bar_nb_all_syngo')


	ensig_hk_labels=['Non-Brain', 'Housekeeping']
	ensig_hk=[syngo_nb, hk_nb]
	pred_df=load_data_functions.load_predicted_ensig_df()

	for i in range(len(ensig_hk)):

		final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, ensig_hk[i], ensig_training)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	
		print (auc)
		
		graph_functions.plot_single_ROC(tpr, fpr, auc, ensig_hk_labels[i])

	#graph_functions.plot_annotate_ROC_controls(tpr, fpr, auc)


