#Goal: compare the indirect edges for brain ppi and HEK cell networks

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt
from itertools import combinations
from statistics import mean

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


from scipy.stats import hypergeom
import sys
sys.path.append('../../read_data_functions/')
import load_data_functions

import sys
sys.path.append('../rwalk_functions/')
import make_brain_ppi_functions
import make_network_graph_functions
import net_random_walk_functions

import sys
sys.path.append('../run_rwalk/')
import run_net_rwalk

import sys
sys.path.append('../graph_functions/')
import graph_functions



def make_histogram(lengths, x_label, y_label, gtitle, name):
	fig = plt.figure()
	n, bins, patches = plt.hist(x=lengths, bins=40, alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(gtitle)
	plt.savefig(name+'.pdf', format="pdf")


if __name__=="__main__":

	#load the big gene pool for main random forest:
	big_pool=load_data_functions.load_big_pool()
	print (len(big_pool))

	#load the bioplex ppi:
	bio_fil=run_net_rwalk.make_filtered_bioplex()
	print (bio_fil)
	bio_kernel=net_random_walk_functions.construct_prop_kernel(bio_fil, 0.4, verbose=True)
	weights=bio_kernel.values.tolist()
	print (len(weights))

	genes=list(bio_kernel.index)
	gene_pairs=list(combinations(genes, 2))
	avg_wts=[]
	for item in gene_pairs:
		weight1=bio_kernel.loc[item[0], item[1]]
		weight2=bio_kernel.loc[item[1], item[0]]
		avg=mean([weight1, weight2])
		avg_wts.append(avg)

	make_histogram(avg_wts, 'edge_weights', 'frequency', 'edge weight distr', 'bioplex_kernel_weights')

	# #load brain ppi:
	# brain_net=make_brain_ppi_functions.make_brain_network()
	# brain_kernel=net_random_walk_functions.construct_prop_kernel(brain_net, 0.4, verbose=True)





