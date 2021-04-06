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

matplotlib.rcParams.update({'font.size': 14})


def plot_bargraph(labels, mean_values, xlabel, ylabel, name):
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, align='center', ecolor='black', capsize=10)

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

labels=['Direct', 'Indirect']
mean_value=[3.75, 2]
xlabel='Shared Edge Type'
ylabel='Fold Enrichment'
name='edge_comparison'
plot_bargraph(labels, mean_values, xlabel, ylabel, name)
