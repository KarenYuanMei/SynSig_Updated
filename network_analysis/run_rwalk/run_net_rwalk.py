#Goal: use the environment jh_network to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import time
import scipy.stats as stats
import sklearn.metrics as metrics

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#import matplotlib.pyplot as plt
import sys
sys.path.append('../rwalk_functions/')
import make_network_graph_functions
import net_random_walk_functions
import net_roc_functions

sys.path.append('../../read_data_functions')
import load_data_functions

sys.path.append('../../graph_functions')
import graph_functions

sys.path.append('../ppi_files/')


def load_bioplex_df():
	filename='../ppi_files/BioPlex 3 - HEK293T default edge.csv'
	df=make_network_graph_functions.make_bioplex_df(filename)
	return df

def load_mentha_df():
	filename='../ppi_files/Human_Mentha_converted.csv'
	df=make_network_graph_functions.make_mentha_df(filename)
	return df

def find_cv_seeds(nodes):
	pos=load_data_functions.get_gene_names('../../run_ML/ML_output/training_genes/updated_positives.csv')
	seeds=list(set(pos)&set(nodes))
	return seeds

def find_cv_nodesets(G, seeds):
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

def sweep_alpha_aucs(G, nodesets, neg):
	alphas=np.arange(0.1, 1, 0.1)

	alpha_cvs={}

	all_mean_aucs=[]
	for item in alphas:
		kernel=net_random_walk_functions.construct_prop_kernel(G, item, verbose=True)
		df=net_random_walk_functions.find_prop_scores_df(kernel, nodesets, 0.8)
		#print (df)
		mean_fpr, tprs, aucs=net_roc_functions.calc_cv_prop_aucs(df, neg)
		alpha_cvs[item]=(tprs, mean_fpr, aucs)
		mean_aucs=np.mean(aucs)
		all_mean_aucs.append(mean_aucs)
	return alpha_cvs, all_mean_aucs

def find_single_alpha_auc(G, nodesets, alpha, neg):
	kernel=net_random_walk_functions.construct_prop_kernel(G, alpha, verbose=True)
	df=net_random_walk_functions.find_prop_scores_df(kernel, nodesets, 0.8)
		#print (df)
	mean_fpr, tprs, aucs=net_roc_functions.calc_cv_prop_aucs(df, neg)
	return mean_fpr, tprs, aucs

def find_opt_alpha(all_mean_aucs):
	alphas=np.arange(0.1, 1, 0.1)
	max_auc=max(all_mean_aucs)
	max_index=all_mean_aucs.index(max_auc)
	opt_alpha=alphas[max_index]
	return opt_alpha	

def find_ordered_set(syngo_nodes, cv_seeds):
	non_seed_pos=list(set(syngo_nodes)-set(cv_seeds))
	ordered_test=cv_seeds+non_seed_pos
	ordered_set={'syngo': ordered_test}
	fraction=len(cv_seeds)/float(len(ordered_test))
	return ordered_set, fraction

def find_syngo_nodes(G):
	syngo=load_data_functions.get_gene_names('../../correct_db/corr_syngo_cc.csv')
	nodes=list(G.nodes())
	syngo_nodes=list(set(nodes)&set(syngo))
	#print ('syngo nodes', len(syngo_nodes))
	return syngo_nodes

def find_hk_nodes(G):
	hk=pd.read_csv('../../gene_lists/Human_Mouse_Common.csv', sep='\t')
	print (hk)
	hk=hk['Human'].tolist()
	nodes=list(G.nodes())
	hk_nodes=list(set(nodes)&set(hk))
	return hk_nodes

def find_net_syngo_test_auc(G,opt_alpha):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	#syngo_nodes=find_syngo_nodes(G)
	#print ('syngo nodes', len(syngo_nodes))
	syngo_nodes=find_hk_nodes(G)

	ordered_set, seed_fraction=find_ordered_set(syngo_nodes, cv_seeds)

	neg=list(set(nodes)-set(syngo_nodes))

	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)
	df=net_random_walk_functions.find_prop_scores_df(kernel, ordered_set, seed_fraction)
	fpr, tpr, threshold, roc_auc=net_roc_functions.calc_net_test_roc(df, neg)
	return fpr, tpr, threshold, roc_auc

def find_net_syngo_shuffled_auc(G, opt_alpha):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	syngo_nodes=find_syngo_nodes(G)
	ordered_set, seed_fraction=find_ordered_set(syngo_nodes, cv_seeds)

	neg=list(set(nodes)-set(syngo_nodes))

	shuff_rocs=net_roc_functions.find_shuff_aucs(G, ordered_set, neg, opt_alpha, seed_fraction, 10)
	print (shuff_rocs)
	return shuff_rocs

def find_deg_matched_auc(G, opt_alpha):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	bg=list(set(nodes)-set(cv_seeds))

	buckets=net_random_walk_functions.make_seed_bg_buckets(G, cv_seeds, bg)
	rand_seeds=net_random_walk_functions.find_rand_samples(G, cv_seeds, buckets)

	syngo_nodes=find_syngo_nodes(G)
	ordered_set=find_ordered_set(syngo_nodes, rand_seeds)

	ordered_set, seed_fraction=find_ordered_set(syngo_nodes, rand_seeds)

	neg=list(set(nodes)-set(syngo_nodes)-set(rand_seeds))
	
	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)
	df=net_random_walk_functions.find_prop_scores_df(kernel, ordered_set, seed_fraction)
	fpr, tpr, threshold, roc_auc=net_roc_functions.calc_net_test_roc(df, neg)
	return roc_auc


if __name__ == '__main__':
	
	net_df=load_bioplex_df()

	G=make_network_graph_functions.make_network_G(net_df)
	print ('orig', len(list(G.nodes())))

	hek_genes=load_data_functions.get_gene_names('../expression_file/hek_genes.csv')

	G=make_network_graph_functions.filter_by_hek_genes(G, hek_genes)
	print ('filtered', len(list(G.nodes())))

	nodes=list(G.nodes())

	cv_seeds=find_cv_seeds(nodes)

	cv_seedsets=find_cv_nodesets(G, cv_seeds)
	#print (cv_seedsets)

	neg=list(set(nodes)-set(cv_seeds))

	#alpha_cvs, all_mean_aucs=sweep_alpha_aucs(G, cv_seedsets, neg)


	#opt_alpha=find_opt_alpha(all_mean_aucs)
	#print (opt_alpha)

	#tprs, mean_fpr, aucs=alpha_cvs[opt_alpha]
	tprs, mean_fpr, aucs=find_single_alpha_auc(G, cv_seedsets, 0.5, neg)
	#print (aucs) #0.6708522690436207

	opt_alpha=0.5
	for i in range(10):
		fpr, tpr, threshold, roc_auc=find_net_syngo_test_auc(G, opt_alpha)
		print (roc_auc)
	graph_functions.plot_single_ROC(tpr, fpr, roc_auc, 'bioplex_hek_only_test')

	#shuff_rocs=find_net_syngo_shuffled_auc(G, opt_alpha)
	#[0.5727682062515527, 0.5565968562656953, 0.5786683737253715, 0.5644656586873242, 0.5735674218383795, 0.5515552861541781, 0.5731787150472272, 0.5642616951976495, 0.5783615678854178, 0.5725117458299364]

	for i in range(10):
		rand_seed_rocs=find_deg_matched_auc(G, opt_alpha)
		print (rand_seed_rocs)

	#[0.5633969110804465, 0.539907829900075, 0.553555791667886, 0.547315431027054, 0.5596714041066664, 0.5569557925772646, 0.5325950565454453, 0.5513386429351921, 0.5593669257034042, 0.5523000393229651]

	
