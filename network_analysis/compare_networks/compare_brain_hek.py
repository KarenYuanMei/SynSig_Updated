#Goal: compare the brain ppi and HEK cell network edges

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt

from scipy.stats import hypergeom
import sys
sys.path.append('../../read_data_functions/')
import load_data_functions

import sys
sys.path.append('../rwalk_functions/')
import make_brain_ppi_functions
import make_network_graph_functions

def filter_nodes(G, nodes_to_keep):
	nodes=list(G.nodes())
	nodes_remove=list(set(nodes)-set(nodes_to_keep))
	G.remove_nodes_from(nodes_remove)
	return G

def find_hypergeometric(genes, pred_no_training):

	overlap=list(set(genes)&set(pred_no_training))
	M=300700
	N=len(genes)
	n=len(pred_no_training)
	x=len(overlap)
	#print ('x', x)

	print ('M', M, 'N', N, 'n', n, 'x', x)
	#x=190
	pval = hypergeom.sf(x-1, M, n, N)

	rv = hypergeom(M, n, N)
	distr = np.arange(0, n+1)
	#print (x)
	prob = rv.pmf(distr)

	maximum=np.max(prob)
	result = np.where(prob == maximum)
	#print (result)
	#result=result.tolist()
	result=result[0]
	#print (result)
	fold=x/float(result)
	fold=fold.tolist()
	print ('Fold Enrichment', fold)
	print ('hypergeometric p-value', pval)
	return fold

if __name__ == '__main__':
	
	brain_net=make_brain_ppi_functions.make_brain_network()
	#print (len(brain_net.edges()))
	brain_nodes=brain_net.nodes()

	bioplex_file='../ppi_files/BioPlex 3 - HEK293T default edge.csv'

	bioplex_df=make_network_graph_functions.make_bioplex_df(bioplex_file)
	bioplex_net=make_network_graph_functions.make_network_G(bioplex_df)
	hek_genes=make_network_graph_functions.find_hek_genes()
	hek_net=make_network_graph_functions.filter_by_hek_genes(bioplex_net, hek_genes)
	#print (len(hek_net.edges()))
	hek_nodes=hek_net.nodes()

	common_nodes=list(set(brain_nodes)&set(hek_nodes()))
	print (len(common_nodes))

	filtered_brain=filter_nodes(brain_net, common_nodes)
	filtered_hek=filter_nodes(hek_net, common_nodes)

	filtered_brain_edges=list(set(filtered_brain.edges()))
	print (len(filtered_brain_edges))
	filtered_hek_edges=list(set(filtered_hek.edges()))
	print (len(filtered_hek_edges))

	shared_edges=list(set(filtered_brain_edges)&set(filtered_hek_edges))
	print (len(shared_edges))



	find_hypergeometric(filtered_brain_edges, filtered_hek_edges)