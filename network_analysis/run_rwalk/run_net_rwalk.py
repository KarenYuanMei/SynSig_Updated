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

	all_mean_aucs=[]
	for item in alphas:
		kernel=net_random_walk_functions.construct_prop_kernel(G, item, verbose=True)
		df=net_random_walk_functions.find_prop_scores_df(kernel, nodesets, 0.8)
		#print (df)
		mean_fpr, tprs, aucs=net_roc_functions.calc_prop_aucs(df, neg)
		mean_aucs=np.mean(aucs)
		all_mean_aucs.append(mean_aucs)
	return all_mean_aucs


def find_opt_alpha(all_mean_aucs):
	alphas=np.arange(0.1, 1, 0.1)
	max_auc=max(all_mean_aucs)
	max_index=all_mean_aucs.index(max_auc)
	opt_alpha=alphas[max_index]
	return opt_alpha	

net_df=load_bioplex_df()

G=make_network_graph_functions.make_network_G(net_df)

nodes=list(G.nodes())

cv_seeds=find_cv_seeds(nodes)

cv_seedsets=find_cv_nodesets(G, cv_seeds)

neg=list(set(nodes)-set(cv_seeds))

all_mean_aucs=sweep_alpha_aucs(G, cv_seedsets, neg)

print (all_mean_aucs)
print (np.mean(all_mean_aucs))

opt_alpha=find_opt_alpha(all_mean_aucs)
print (opt_alpha)

# syngo=load_data_functions.get_gene_names('../../correct_db/corr_syngo_cc.csv')


# #

# #

# #print (df)


# #print (G.number_of_edges())



# #print (len(pos))

# #print (len(seeds))

# #print (nodesets)

# #
# #



# print (len(syngo))


# overlap=list(set(nodes)&set(syngo))
# print ('overlap', len(overlap))

# fraction=len(seeds)/float(len(overlap))
# print (len(seeds)/float(len(overlap)))

# non_seeds=list(set(syngo)-set(seeds))
# ordered_test=seeds+non_seeds
# ordered_set={'syngo': ordered_test}

# neg=list(set(nodes)-set(syngo))

# kernel=net_random_walk_functions.construct_prop_kernel(G, 0.4, verbose=True)
# df=find_prop_scores_df(kernel, ordered_set, fraction)
# #print (df)
# 		#print (df)
# cols=['Sub-Sample', 'Non-Sample', 'Prop Score']
# subdf=df[cols]
# fpr, tpr, threshold, roc_auc=calculate_roc(subdf, neg)
# print (roc_auc)

# shuff_rocs=find_shuff_scores(G, ordered_set, 0.4, fraction, 10)
# print (shuff_rocs)



