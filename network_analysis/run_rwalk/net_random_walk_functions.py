import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import time

def set_p(nodesets, fraction):
	#network_nodes = [str(gene) for gene in network.nodes()]
	nodesets_p = {}
	for nodeset in nodesets:
		#nodesets_coverage = len([node for node in nodesets[nodeset] if node in network_nodes])
		#print (nodesets_coverage)
		nodesets_p[nodeset] = fraction
	#print (nodesets_p)
	return nodesets_p

# Normalize network (or network subgraph) for random walk propagation
def normalize_network(network, symmetric_norm=False):
	adj_mat = nx.adjacency_matrix(network)
	adj_array = np.array(adj_mat.todense())
	if symmetric_norm:
		D = np.diag(1/np.sqrt(sum(adj_array)))
		adj_array_norm = np.dot(np.dot(D, adj_array), D)
	else:
		degree_norm_array = np.diag(1/sum(adj_array).astype(float))
		#print (degree_norm_array)
		sparse_degree_norm_array = sp.sparse.csr_matrix(degree_norm_array)
		adj_array_norm = sparse_degree_norm_array.dot(adj_mat).toarray()
	return adj_array_norm

# Closed form random-walk propagation (as seen in HotNet2) for each subgraph: Ft = (1-alpha)*Fo * (I-alpha*norm_adj_mat)^-1
# Concatenate to previous set of subgraphs
def fast_random_walk(alpha, binary_mat, subgraph_norm, prop_data):
	term1=(1-alpha)*binary_mat
	term2=np.identity(binary_mat.shape[1])-alpha*subgraph_norm
	term2_inv = np.linalg.inv(term2)
	subgraph_prop = np.dot(term1, term2_inv)
	return np.concatenate((prop_data, subgraph_prop), axis=1)


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

def construct_prop_kernel(network, alpha, verbose=False, save_path=None):
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

# Wrapper to calculate AUPRC of multiple node sets' recovery for large networks (>=250k edges)
def get_propagated_scores(net_kernel, genesets, genesets_p, n=1, cores=1, bg=None, verbose=True):
	starttime = time.time()
	# Construct binary gene set sub-sample matrix
	geneset_list = genesets.keys()
	m, c = len(geneset_list), net_kernel.shape[0]
	subsample_mat = np.zeros((n*m, c))
	y_actual_mat = np.zeros((n*m, c))
	# Each block of length n rows is a sub-sampled binary vector of the corresponding gene set
	for i in range(m):
		geneset = geneset_list[i]
		# Get indices of gene set genes in kernel
		intersect = [gene for gene in genesets[geneset] if gene in net_kernel.index]
		index_dict = dict((gene, idx) for idx, gene in enumerate(net_kernel.index))
		intersect_idx = [index_dict[gene] for gene in intersect]
		
		# Sub-sample gene set indices
		sample_size = int(round(genesets_p[geneset]*len(intersect)))
		#sample_idx = random.sample(intersect_idx, sample_size)
		sample_idx=intersect_idx[:sample_size]
		non_sample_idx = [idx for idx in intersect_idx if idx not in sample_idx]
		# Set sub-sampled list to 1
		row = i

		subsample_mat[row, sample_idx] = 1
		y_actual_mat[row, non_sample_idx] = 1
	if verbose:
		print ('Binary gene set sub-sample matrix constructed')
	# Propagate sub-samples
	prop_subsamples = np.dot(subsample_mat, net_kernel)
	if verbose:
		print ('Binary gene set sub-sample matrix propagated')
	# Construct parameter list to be passed
	
	final_scores=[]
	for i in range(len(geneset_list)):
		
		row = i
		prop_result_full = pd.DataFrame(np.array((subsample_mat[row], y_actual_mat[row], prop_subsamples[row])), 
								   		index=['Sub-Sample', 'Non-Sample', 'Prop Score'], columns=net_kernel.columns).T
		#prop_result_full.to_csv('%s_prop_result_full_%s.csv'%(name, row))
		final_scores.append(prop_result_full)

	all_scores=pd.concat(final_scores, axis=1)

	return all_scores

