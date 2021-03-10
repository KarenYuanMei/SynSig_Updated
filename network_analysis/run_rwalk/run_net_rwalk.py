#Goal: use the environment jh_network to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import time
import scipy.stats as stats
import sklearn.metrics as metrics

#import matplotlib.pyplot as plt
import sys
sys.path.append('../rwalk_functions/')
import make_network_graph_functions
import net_random_walk_functions
import net_roc_functions

sys.path.append('../../read_data_functions')
import load_data_functions

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
		alpha_cvs[item]=(aucs)
		mean_aucs=np.mean(aucs)
		all_mean_aucs.append(mean_aucs)
	return alpha_cvs, all_mean_aucs


def find_opt_alpha(all_mean_aucs):
	alphas=np.arange(0.1, 1, 0.1)
	max_auc=max(all_mean_aucs)
	max_index=all_mean_aucs.index(max_auc)
	opt_alpha=alphas[max_index]
	return opt_alpha	


def find_net_syngo_test_auc(G, opt_alpha):
	syngo=load_data_functions.get_gene_names('../../correct_db/corr_syngo_cc.csv')
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	syngo_nodes=list(set(nodes)&set(syngo))
	seed_fraction=len(cv_seeds)/float(len(syngo_nodes))
	print (seed_fraction)

	non_seed_pos=list(set(syngo)-set(cv_seeds))
	ordered_test=cv_seeds+non_seed_pos
	ordered_set={'syngo': ordered_test}

	neg=list(set(nodes)-set(syngo))

	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)
	df=find_prop_scores_df(kernel, ordered_set, seed_fraction)
	roc_auc=calc_net_test_roc(df)
	return roc_auc

hek_genes=load_data_functions.get_gene_names('../expression_file/hek_genes.csv')


net_df=load_bioplex_df()

G=make_network_graph_functions.make_network_G(net_df)
print ('orig', len(list(G.nodes())))

G=make_network_graph_functions.filter_by_hek_genes(G, hek_genes)
print ('filtered', len(list(G.nodes())))

nodes=list(G.nodes())

cv_seeds=find_cv_seeds(nodes)

cv_seedsets=find_cv_nodesets(G, cv_seeds)
#print (cv_seedsets)

neg=list(set(nodes)-set(cv_seeds))

alpha_cvs, all_mean_aucs=sweep_alpha_aucs(G, cv_seedsets, neg)

print (alpha_cvs)
print (all_mean_aucs)

opt_alpha=find_opt_alpha(all_mean_aucs)
print (opt_alpha)

roc_auc=find_net_syngo_test_auc(G, opt_alpha)
shuff_rocs=net_roc_functions.find_shuff_aucs(G, ordered_set, opt_alpha, seed_fraction, 10)
print (shuff_rocs)

