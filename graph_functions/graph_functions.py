#Goal: make graphs

import numpy as np
from igraph import *
import pandas as pd
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#-----load your ontology onto HiView-------------------------------------------------------------------------
#code for uploading to HiView taken from DDOT package: https://github.com/michaelkyu/ddot/blob/master/examples/Tutorial.ipynb

import networkx as nx

import matplotlib
#matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

#import seaborn as sns; sns.set()

import pylab

#construct the random forest so that when doing the 5X cross validation, the model is not seeing 20% of the genes, not just rows--------------------
import random
import pickle
#from define_gene_objects_rf_5 import PairOfGenes

from sklearn.metrics import auc
from itertools import combinations
from collections import defaultdict

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
matplotlib.rcParams.update({'font.size': 14})


def plot_mean_ROC(tprs, mean_fpr, auc_list):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')

	mean_auc=np.mean(auc_list)

	mean_tpr = np.mean(tprs, axis=0)

	plt.plot(mean_fpr, mean_tpr, color='b',
	         label=r'ROC (AUC = %0.3f)' % (mean_auc),
	         lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
	                 label=r'$\pm$ 1 std. dev.')
	#plt.title('Avg ROC Curve for Predicting Synapse Genes \n 5-Fold Cross-Validation', fontweight = 'bold')

	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')
	plt.grid(False)
	# show the legend
	plt.legend()
		# show the plot
	#plt.show()
	plt.savefig('updated_val_5fold_ROC.svg', format="svg")


def plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel):
	x_pos=np.arange(len(labels))
	plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#557f2d','silver', 'dimgray','#2d7f5e'], align='center', ecolor='black', capsize=10)

	plt.ylim(0, 1)
	# Create legend & Show graphic
	#plt.legend()
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	plt.xticks(rotation=45)
	plt.savefig('compare_regressor_roc.svg', format="svg")
