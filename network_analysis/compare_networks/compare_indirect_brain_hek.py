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

def find_prop_edges_wt_df(gene_pairs, avg_wts, name):
	node1=[x[0] for x in gene_pairs]
	node2=[x[1] for x in gene_pairs]

	df=pd.DataFrame({'Node1': node1, 'Node2': node2, 'weights': avg_wts})
	df.to_csv('%s_prop_edge_wts.csv'%name)
	return df

def load_prop_edges_wt_df(df_name):
	df=pd.read_csv('%s_prop_edge_wts.csv'%df_name, index_col=[0])
	return df

if __name__=="__main__":

	# #load the big gene pool for main random forest:
	# big_pool=load_data_functions.load_big_pool()
	# print (len(big_pool))

	# #load the bioplex ppi:
	# bio_fil=run_net_rwalk.make_filtered_bioplex()
	# #print (bio_fil)
	# bio_kernel=net_random_walk_functions.construct_prop_kernel(bio_fil, 0.4, verbose=True)
	# #gene_pairs, avg_wts=find_avg_wts(bio_kernel)
	# #prop_df=find_prop_edges_wt_df(gene_pairs, avg_wts, 'bio_fil')

	# #new=bio_kernel.stack().reset_index().rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})
	# #new.to_csv('biofil_prop_edge_wts.csv'


	# # #load brain ppi:
	# brain_ppi=pd.read_csv('../../source_data_files/ppi_files/brain_ppi_df.csv')
	# brain_net=make_network_graph_functions.make_network_G(brain_ppi)
	# brain_kernel=net_random_walk_functions.construct_prop_kernel(brain_net, 0.4, verbose=True)
	# #brain_prop_df=brain_kernel.stack().reset_index().rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})
	# #brain_prop_df.to_csv('brain_prop_edge_wts.csv')

	# common_nodes=list(set(bio_fil.nodes())&set(brain_net.nodes()))
	# print (len(common_nodes))
	# gene_pairs=list(combinations(common_nodes, 2))
	# print (len(gene_pairs))

	# gene_pairs, avg_wts=find_avg_wts(common_nodes, bio_kernel)
	# find_prop_edges_wt_df(gene_pairs, avg_wts, 'bio_fil')

	# gene_pairs, avg_wts=find_avg_wts(common_nodes, brain_kernel)
	# find_prop_edges_wt_df(gene_pairs, avg_wts, 'brain_net')

	bio_fil=load_prop_edges_wt_df('bio_fil')
	print (bio_fil)

	brain_net=load_prop_edges_wt_df('brain_net')
	print (brain_net)









