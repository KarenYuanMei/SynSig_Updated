#Goal: use the environment jh_network to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.append('../rwalk_functions/')
import make_network_graph_functions

sys.path.append('../../read_data_functions')
import load_data_functions


sys.path.append('../ppi_files/')

filename='../ppi_files/Human_Mentha_converted.csv'
df=make_network_graph_functions.make_mentha_df(filename)
#print (df)

G=make_network_graph_functions.make_network_G(df)
print (G.number_of_edges())
nodes=list(G.nodes())

pos=load_data_functions.get_gene_names('../../run_ML/ML_output/training_genes/updated_positives.csv')
print (len(pos))

seeds=list(set(pos)&set(nodes))
print (len(seeds))

def find_nodesets(G, seeds):
	no_per_fold=len(seeds)/5
	keys=['first', 'second', 'third', 'fourth', 'fifth']
	seed_lists = [seeds[x:x+no_per_fold] for x in range(0, len(seeds), no_per_fold)]
	val_lists=[]
	for item in seed_lists:
		val_list=list(set(seeds)-set(item))
		val_lists.append(val_list)
	ordered=[]
	for i in range(len(seed_lists)):
		each=seed_lists[i]+val_lists[i]
		ordered.append(each)
	nodesets=dict(zip(keys, ordered))
	return nodesets

nodesets=find_nodesets(G, seeds)
#print (nodesets)

neg=load_data_functions.get_gene_names('../../run_ML/ML_output/training_genes/updated_negatives.csv')
print (len(neg))
neg=list(set(neg)&set(nodes))
print (len(neg))

def calculate_p(nodesets):
	#network_nodes = [str(gene) for gene in network.nodes()]
	nodesets_p = {}
	for nodeset in nodesets:
		#nodesets_coverage = len([node for node in nodesets[nodeset] if node in network_nodes])
		#print (nodesets_coverage)
		nodesets_p[nodeset] = 0.2
	print (nodesets_p)
	return nodesets_p

nodesets_p=calculate_p(nodesets)
print (nodesets_p)

# Wrapper for random walk propagation of full network by subgraphs
def closed_form_network_propagation(network, binary_matrix, network_alpha, symmetric_norm=False,  verbose=False, save_path=None):
	starttime=time.time()
	if verbose:
		print ('Alpha:', network_alpha)
	# Separate network into connected components and calculate propagation values of each sub-sample on each connected component
	subgraphs = list(nx.connected_component_subgraphs(network))
	#subgraphs=list(nx.subgraph(c) for c in connected_components(G))
	#subgraphs=list(G.subgraph(c) for c in nx.connected_components(G))
	# Initialize propagation results by propagating first subgraph
	subgraph = subgraphs[0]
	subgraph_nodes = list(subgraph.nodes)
	prop_data_node_order = list(subgraph_nodes)
	binary_matrix_filt = np.array(binary_matrix.T.loc[subgraph_nodes].fillna(0).T)
	subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
	prop_data_empty = np.zeros((binary_matrix_filt.shape[0], 1))
	prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm, prop_data_empty)
	# Get propagated results for remaining subgraphs
	for subgraph in subgraphs[1:]:
		subgraph_nodes = list(subgraph.nodes)
		prop_data_node_order = prop_data_node_order + subgraph_nodes
		binary_matrix_filt = np.array(binary_matrix.T.loc[subgraph_nodes].fillna(0).T)
		subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
		prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm, prop_data)
	# Return propagated result as dataframe
	prop_data_df = pd.DataFrame(data=prop_data[:,1:], index = binary_matrix.index, columns=prop_data_node_order)
	if save_path is None:
		if verbose:
			print ('Network Propagation Complete:', time.time()-starttime, 'seconds')		
		return prop_data_df
	else:
		prop_data_df.to_csv(save_path)
		if verbose:
			print ('Network Propagation Complete:', time.time()-starttime, 'seconds')				
		return prop_data_df

def construct_prop_kernel(network, alpha, m=-0.02935302, b=0.74842057, verbose=False, save_path=None):
	network_Fo = pd.DataFrame(data=np.identity(len(network.nodes())), index=network.nodes(), columns=network.nodes())
	alpha_val = alpha
	network_Fn = closed_form_network_propagation(network, network_Fo, alpha_val, verbose=verbose)
	network_Fn = network_Fn.loc[network_Fn.columns]
	if verbose:
		print ('Propagated network kernel constructed')
	if save_path is not None:
		if save_path.endswith('.hdf'):
			network_Fn.to_hdf(save_path, key='Kernel', mode='w')
		else:
			network_Fn.to_csv(save_path)
	return network_Fn

network_fn=construct_prop_kernel(G, 0.4)
print (network_fn)