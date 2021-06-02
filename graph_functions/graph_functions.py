#Goal: make graphs

import numpy as np
#from igraph import *
import pandas as pd
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#-----load your ontology onto HiView-------------------------------------------------------------------------
#code for uploading to HiView taken from DDOT package: https://github.com/michaelkyu/ddot/blob/master/examples/Tutorial.ipynb

import networkx as nx

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles

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
	plt.xlim([0, 1])
	plt.ylim([0, 1])

	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')
	plt.grid(False)
	# show the legend
	plt.legend()
		# show the plot
	#plt.show()
	plt.savefig('%s_5fold_ROC.svg'%name, format="svg")

def plot_single_ROC(tpr, fpr, auc, plotcolor, name):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')


	plt.plot(fpr, tpr, color=plotcolor,
	         label=r'ROC (AUC = %0.2f)' % (auc),
	         lw=2, alpha=.8)

	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')
	plt.grid(False)
	# show the legend
	plt.legend()
	plt.xlim([0, 1])
	plt.ylim([0, 1])
		# show the plot
	#plt.show()
	plt.savefig('%s_ROC.svg'%name, format="svg")

def plot_tandem_ROC(tpr, fpr, auc, name):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')


	plt.plot(fpr, tpr,
	         label=r'ROC (AUC = %0.2f)' % (auc),
	         lw=2, alpha=.8)

	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')
	plt.grid(False)
	# show the legend
	plt.legend()
	plt.xlim([0, 1])
	plt.ylim([0, 1])
		# show the plot
	#plt.show()
	plt.savefig('%s_ROC.svg'%name, format="svg")

def plot_annotate_ROC_controls(tpr, fpr, auc):
	plt.plot([0,1],[0,1],linestyle = '--',color = 'black', label='Random Chance')

	plt.plot(fpr, tpr, color='maroon',
	         label=r'ROC (AUC = %0.2f)' % (auc),
	         lw=2, alpha=.8)

	#without training genes: {'hk': (0.16326530612244897, 0.0719837783034809), 'golgi': (0.29591836734693877, 0.08122124634044332), 'mem': (0.21428571428571427, 0.130806362269526)}

	plt.plot(0.11, 0.7755, color='purple', marker='o', markersize=10)
	plt.annotate('SynSig', color='purple', xy=(0.11, 0.7755), xytext=(0.11+0.05, 0.7755), arrowprops=dict(facecolor='purple', lw=2, arrowstyle='->'))

	plt.plot(0.08, 0.125, color='#7f6d5f', marker='o', markersize=10)
	plt.annotate('Housekeeping', color='#7f6d5f', xy=(0.08, 0.125), xytext=(0.08+0.05, 0.125), arrowprops=dict(facecolor='#7f6d5f', lw=2, arrowstyle='->'))

	plt.plot(0.09, 0.23, color='#557f2d', marker='o', markersize=10)
	plt.annotate('Golgi App.', color='#557f2d', xy=(0.09, 0.23), xytext=(0.09+0.05, 0.23), arrowprops=dict(facecolor='#557f2d', lw=2, arrowstyle='->'))

	plt.plot(0.14, 0.23, color='silver', marker='o', markersize=10)
	plt.annotate('Transmem.', color='silver', xy=(0.14, 0.23), xytext=(0.14+0.05, 0.23), arrowprops=dict(facecolor='silver', lw=2, arrowstyle='->'))


	plt.xlabel('1-Specificity', fontweight='bold')
	plt.ylabel('Sensitivity', fontweight='bold')


	plt.grid(False)
	# show the legend
	plt.legend()
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.savefig('annotate_ROC_controls.svg', format="svg")


def plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, name):
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	#plt.bar(labels, mean_values, yerr=sem, color=['#2d7f5e', '#7f6d5f', '#557f2d','silver'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, yerr=sem, color=['#2d7f5e', '#7f6d5f', '#557f2d'], align='center', ecolor='black', capsize=10)

	plt.ylim(0.5, 1)
	#plt.ylim(1, 10**5)
	#plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	plt.xticks(rotation=45)
	plt.savefig(name+'.svg', format="svg")

def plot_alpha(df, name):
	f = plt.figure()
	x=np.arange(1,10,1)
	plt.errorbar(x, df['mean'].tolist(), yerr=df['sem'].tolist())

	plt.xlabel('Alpha')
	plt.ylabel('Avg Cross-Validation ROC AUC')
	plt.title ("Optimizing Propagation Constant Alpha", fontweight='bold')
	x_labels = list(df.index)
	plt.xticks(x, x_labels)

	#y_ticks = np.arange(0, 1, 0.1)
	#plt.yticks(y_ticks)
	plt.ylim(0, 1)
	plt.show()
	plt.close()
	f.savefig("%s_sweep_alpha.svg"%name, bbox_inches='tight')

def plot_bargraph(labels, mean_values, xlabel, ylabel, name):
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, align='center', color='#2d7f5e', ecolor='black', capsize=10)

	#plt.ylim(1, 10**5)
	#plt.ylim(0.5, 1)
	#plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	#y_ticks = np.arange(0, 25, 5)
	#plt.yticks(y_ticks)
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	#plt.xticks(rotation=45)
	plt.savefig(name+'.svg', format="svg")
	plt.close()

def plot_linegraph(x_values, y_values, xlabel, ylabel, name):
	plt.plot(x_values, y_values)
	plt.ylabel(ylabel)
	plt.ylim(0.5, 0.9)
	plt.xlabel(xlabel)
	plt.savefig(name+'.svg', format="svg")
	plt.close()

def plot_venn2(genelist1, genelist2, labels, name):
	venn2([set(genelist1), set(genelist2)], set_labels = (labels[0], labels[1]))
	plt.savefig(name+'.svg', format="svg")
	plt.close()

def make_histogram(lengths, x_label, y_label, gtitle, name):
	fig = plt.figure()
	n, bins, patches = plt.hist(x=lengths, bins=40, color='#0504aa',
	                            alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(gtitle)
	plt.savefig(name+'.pdf', format="pdf")