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

import compare_brain_hek

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
	n, bins, patches = plt.hist(x=lengths, bins=40, alpha=0.7, rwidth=0.85, histtype='step')
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(gtitle)
	plt.savefig(name+'.pdf', format="pdf")

def find_avg_wts(common_nodes, kernel):
	gene_pairs=list(combinations(common_nodes, 2))
	print (len(gene_pairs))
	avg_wts=[]
	for item in gene_pairs:
		weight1=kernel.loc[item[0], item[1]]
		weight2=kernel.loc[item[1], item[0]]
		avg=mean([weight1, weight2])
		#print (avg)
		avg_wts.append(avg)
	return gene_pairs, avg_wts

def find_prop_edges_wt_df(common_nodes, kernel, name):
	gene_pairs, avg_wts=find_avg_wts(common_nodes, kernel)
	node1=[x[0] for x in gene_pairs]
	node2=[x[1] for x in gene_pairs]

	df=pd.DataFrame({'Node1': node1, 'Node2': node2, 'weights': avg_wts})
	df.to_csv('%s_prop_edge_wts.csv'%name)
	return df

def load_prop_edges_wt_df(df_name):
	df=pd.read_csv('%s_prop_edge_wts.csv'%df_name, index_col=[0])
	return df

def threshold_df(df, threshold):
	weights=df['weights'].tolist()
	weights=np.array(weights)
	p=np.percentile (weights, threshold)
	print (p)

	new=df[df['weights']>p]
	return new

if __name__=="__main__":

	#load the bioplex ppi:
	bio_fil=run_net_rwalk.make_filtered_bioplex()
	#print (bio_fil)
	bio_kernel=net_random_walk_functions.construct_prop_kernel(bio_fil, 0.4, verbose=True)

	# #load brain ppi:
	brain_ppi=pd.read_csv('../../source_data_files/ppi_files/brain_ppi_df.csv')
	brain_net=make_network_graph_functions.make_network_G(brain_ppi)
	brain_kernel=net_random_walk_functions.construct_prop_kernel(brain_net, 0.4, verbose=True)

	common_nodes=list(set(bio_fil.nodes())&set(brain_net.nodes()))
	print (len(common_nodes))
	
	bio_fil=find_prop_edges_wt_df(common_nodes, bio_kernel, 'bio_fil')
	brain_net=find_prop_edges_wt_df(common_nodes, brain_kernel, 'brain_net')

	#bio_fil=load_prop_edges_wt_df('bio_fil')
	#print (bio_fil)

	#brain_net=load_prop_edges_wt_df('brain_net')
	#print (brain_net)

	percentiles=[80, 85, 90, 95, 99]
	for item in percentiles:
		bio_fil_th=threshold_df(bio_fil, item)
		print ('bio edges', bio_fil_th.shape[0])
		new_biofil=make_network_graph_functions.make_network_G(bio_fil_th)
		new_biofil_edges=list(new_biofil.edges())

		brain_net_th=threshold_df(brain_net, item)
		print ('brain edges', brain_net_th.shape[0])
		new_brainnet=make_network_graph_functions.make_network_G(brain_net_th)
		new_brainnet_edges=list(new_brainnet.edges())

		shared=list(set(new_biofil_edges)&set(new_brainnet_edges))
		print ('shared', len(shared))

		compare_brain_hek.find_hypergeometric(new_biofil_edges, new_brainnet_edges)









