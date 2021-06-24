#Goal: 1) calculate the ROC of the predicted scores for predicting the synapse genes from SynGO, SynDB, and SynSysNet, and consensus syn
#		this is Figure 2A
#.     2) print the thresholds, tpr, fpr for each evaluation into csv
#      3) compare the sensitivity, specificity of control genes and plot the comparisons: Figure 2D
import pandas as pd
import numpy as np
import csv

import ddot
from ddot import Ontology

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 14})


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


# def plot_tandem_ROC(tpr, fpr, auc, name):
# 	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')


# 	plt.plot(fpr, tpr,
# 	         label=r'ROC (AUC = %0.2f)' % (auc),
# 	         lw=2, alpha=.8)

# 	plt.xlabel('1-Specificity', fontweight='bold')
# 	plt.ylabel('Sensitivity', fontweight='bold')
# 	plt.grid(False)
# 	# show the legend
# 	plt.legend()
# 	plt.xlim([0, 1])
# 	plt.ylim([0, 1])
# 		# show the plot
# 	#plt.show()
# 	plt.savefig('%s_ROC.svg'%name, format="svg")

def plot_annotate_ROC_controls(tpr, fpr, auc):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')

	plt.plot(fpr, tpr, color='maroon',
	         label=r'ROC (AUC = %0.2f)' % (auc),
	         lw=2, alpha=.8)

	#without training genes: (tpr, fpr): {'hk': (0.16326530612244897, 0.0719837783034809), 
	#'nuclear': (0.1836734693877551 0.27760605064123645)
	#'golgi': (0.29591836734693877, 0.08122124634044332), 

	#'mem': (0.21428571428571427, 0.130806362269526)}

	plt.plot(0.11, 0.7755, color='purple', marker='o', markersize=10)
	plt.annotate('SynSig', color='purple', xy=(0.11, 0.7755), xytext=(0.11+0.05, 0.7755), arrowprops=dict(facecolor='purple', lw=2, arrowstyle='->'))

	plt.plot(0.072, 0.16, color='#7f6d5f', marker='o', markersize=10)
	plt.annotate('Housekeeping', color='#7f6d5f', xy=(0.072, 0.16), xytext=(0.072+0.05, 0.16), arrowprops=dict(facecolor='#7f6d5f', lw=2, arrowstyle='->'))

	plt.plot(0.2776, 0.18, color='#3776ab', marker='o', markersize=10)
	plt.annotate('Nuclear', color='#3776ab', xy=(0.2776, 0.18), xytext=(0.2776+0.05, 0.18), arrowprops=dict(facecolor='#3776ab', lw=2, arrowstyle='->'))

	plt.plot(0.08, 0.296, color='#557f2d', marker='o', markersize=10)
	plt.annotate('Golgi App.', color='#557f2d', xy=(0.08, 0.296), xytext=(0.08+0.05, 0.296), arrowprops=dict(facecolor='#557f2d', lw=2, arrowstyle='->'))

	plt.plot(0.12, 0.21, color='silver', marker='o', markersize=10)
	plt.annotate('Transmem.', color='silver', xy=(0.12, 0.21), xytext=(0.12+0.05, 0.21), arrowprops=dict(facecolor='silver', lw=2, arrowstyle='->'))


	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')


	plt.grid(False)
	# show the legend
	plt.legend()
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig('annotate_ROC_controls.svg', format="svg")


if __name__ == '__main__':
	
	big_pool=load_data_functions.load_big_pool()

	all_training=find_training_genes_functions.load_pos_neg_training()

	human_ont=find_GO_scores.find_GO_ont()
	go_genes=human_ont.genes

	syngo=load_data_functions.find_syngo(big_pool, go_genes)
	syndb=load_data_functions.find_SynDB(big_pool)
	synsysnet=load_data_functions.find_synsysnet(big_pool)
	syn=list(set(syngo)&set(syndb)&set(synsysnet))

	db_list=[syngo, syndb, synsysnet, syn]
	db_labels=['syngo', 'syndb', 'synsysnet', 'syn']

	pred_df=load_data_functions.load_predicted_synsig_df()

	for i in range(len(db_list)):
		final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, db_list[i], all_training)
		fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)
		print (auc)
		ROC_functions.save_roc_df(thresholds, tpr, fpr, i, db_labels[i])

		#make Figure 2A
		graph_functions.plot_tandem_ROC(tpr, fpr, auc,'tandem')

	final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, syn, all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)	

	graph_functions.plot_single_ROC(tpr, fpr, auc, 'maroon', 'syn')
	print ('tpr fpr auc', tpr, fpr, auc)

	#make Figure 2D
	plot_annotate_ROC_controls(tpr, fpr, auc)


