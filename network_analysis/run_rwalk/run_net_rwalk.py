#Goal: use the environment jh_network to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import time
import scipy.stats as stats
import sklearn.metrics as metrics
import random

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

def make_sweep_alpha_df(alpha_cvs, all_mean_aucs):
	alphas=np.arange(0.1, 1, 0.1)

	sems=[]
	aucs_list=[]
	for key in alpha_cvs:
		tprs, mean_fpr, aucs=alpha_cvs[key]
		aucs_list.append(aucs)
		sem=stats.sem(aucs)
		sems.append(sem)

	df=pd.DataFrame({'alphas': alphas, 'aucs': aucs_list, 'mean': all_mean_aucs, 'sem': sems})
	df=df.set_index('alphas')
	return df

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

def find_synapse_nodes(G):
	syngo=load_data_functions.get_gene_names('../../correct_db/corr_syngo_cc.csv')
	syndb=load_data_functions.get_gene_names('../../correct_db/SynDB.csv')
	synsysnet=load_data_functions.get_gene_names('../../correct_db/synsysnet.csv')

	synapse=list(set(syngo)&set(syndb)&set(synsysnet))

	nodes=list(G.nodes())
	synapse_nodes=list(set(nodes)&set(synapse))
	return synapse_nodes

def find_hk_nodes(G):
	hk=pd.read_csv('../../gene_lists/Human_Mouse_Common.csv', sep=';')
	print (hk)
	hk=hk['Human'].tolist()
	nodes=list(G.nodes())
	hk_nodes=list(set(nodes)&set(hk))
	return hk_nodes

def find_net_test_auc(G,opt_alpha, gold_standards):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)

	if gold_standards=='syngo':
		pos_nodes=find_syngo_nodes(G)
		print ('syngo nodes', len(pos_nodes))
	elif gold_standards=='hk':
		pos_nodes=find_hk_nodes(G)
	elif gold_standards=='synapse':
		pos_nodes=find_synapse_nodes(G)

	#syngo_nodes=find_hk_nodes(G)

	ordered_set, seed_fraction=find_ordered_set(pos_nodes, cv_seeds)

	neg=list(set(nodes)-set(pos_nodes))

	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)
	df=net_random_walk_functions.find_prop_scores_df(kernel, ordered_set, seed_fraction)
	fpr, tpr, threshold, roc_auc=net_roc_functions.calc_net_test_roc(df, neg)
	return fpr, tpr, threshold, roc_auc

def find_net_syngo_shuffled_auc(G, opt_alpha, iterations):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	syngo_nodes=find_syngo_nodes(G)
	ordered_set, seed_fraction=find_ordered_set(syngo_nodes, cv_seeds)

	neg=list(set(nodes)-set(syngo_nodes))

	shuff_rocs=net_roc_functions.find_shuff_aucs(G, ordered_set, neg, opt_alpha, seed_fraction, iterations)
	#print (shuff_rocs)
	return shuff_rocs

def find_deg_matched_auc(G, opt_alpha, kernel, buckets):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	

	rand_seeds=net_random_walk_functions.find_rand_samples(G, cv_seeds, buckets)
	print ('newrand', rand_seeds)

	syngo_nodes=find_syngo_nodes(G)

	ordered_set, seed_fraction=find_ordered_set(syngo_nodes, rand_seeds)

	neg=list(set(nodes)-set(syngo_nodes)-set(rand_seeds))
	
	df=net_random_walk_functions.find_prop_scores_df(kernel, ordered_set, seed_fraction)
	fpr, tpr, threshold, roc_auc=net_roc_functions.calc_net_test_roc(df, neg)
	print (roc_auc)
	return roc_auc

def df_to_network(name):
	if name == 'mentha':
		net_df=load_mentha_df()
		G=make_network_graph_functions.make_network_G(net_df)
		print ('orig', len(list(G.nodes())))

	if name == 'bioplex':
		net_df=load_bioplex_df()
		hek_genes=load_data_functions.get_gene_names('../expression_file/hek_genes.csv')
		G=make_network_graph_functions.make_network_G(net_df)
		print ('orig', len(list(G.nodes())))
		G=make_network_graph_functions.filter_by_hek_genes(G, hek_genes)
		print ('filtered', len(list(G.nodes())))
	return G

def calc_plot_opt_alpha(G, net):
	nodes=list(G.nodes())
	#find the syngo seeds for propagating
	cv_seeds=find_cv_seeds(nodes)
	cv_seedsets=find_cv_nodesets(G, cv_seeds)
	neg=list(set(nodes)-set(cv_seeds))
	alpha_cvs, all_mean_aucs=sweep_alpha_aucs(G, cv_seedsets, neg)
	alpha_df=make_sweep_alpha_df(alpha_cvs, all_mean_aucs)
	alpha_df.to_csv('%s_alpha_df.csv'%net)
	alpha_df=pd.read_csv('%s_alpha_df.csv'%net, index_col=[0])
	graph_functions.plot_alpha(alpha_df, net)
	print ('done')
	opt_alpha=find_opt_alpha(all_mean_aucs)
	return alpha_cvs, opt_alpha

def find_test_auc(net):
	genelists=['syngo', 'hk', 'synapse']

	auc_list=[]
	for item in genelists:
		fpr, tpr, threshold, roc_auc=find_net_test_auc(G, opt_alpha, item)
		graph_functions.plot_single_ROC(tpr, fpr, roc_auc, '%s_%s_test'%(net, item))
		auc_list.append(roc_auc)
	return auc_list

def find_net_deg_marched_auc_list(G, opt_alpha, iterations):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	bg=list(set(nodes)-set(cv_seeds))
	buckets=net_random_walk_functions.make_seed_bg_buckets(G, cv_seeds, bg)
	print ('newbuckets', buckets)
	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)
	all_rand_rocs=[]
	for i in range(iterations):
		rand_seed_rocs=find_deg_matched_auc(G, opt_alpha, kernel, buckets)
		#print (rand_seed_rocs)
		all_rand_rocs.append(rand_seed_rocs)
	print (net, all_rand_rocs)
	return all_rand_rocs

def plot_test_control_aucs(net, auc_list, shuff_rocs, rand_rocs):
	shuff_mean=np.mean(shuff_rocs)
	rand_mean=np.mean(rand_rocs)
	syngo_auc=auc_list[0]

	mean_values=[syngo_auc, shuff_mean, rand_mean]

	shuff_sem=stats.sem(shuff_rocs)
	rand_sem=stats.sem(rand_rocs)
	sem=[0, shuff_sem, rand_sem]

	labels=['SynGO', 'Shuff Net', 'Random Seeds']
	xlabel='Gene Categories'
	ylabel='Recovery ROC'

	graph_functions.plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, net)

def plot_all_test_aucs(net, auc_list):
	labels=['SynGO', 'Housekeeping', 'Synapse']
	xlabel='Gene Categories'
	ylabel='Recovery ROC'

	graph_functions.plot_bargraph(labels, mean_values, xlabel, ylabel, net)

if __name__ == '__main__':

	net_names=['mentha', 'bioplex']
	for net in net_names:

		G=df_to_network(net)

		alpha_cvs, opt_alpha=calc_plot_opt_alpha(G, net)

		print ('opt_alpha', opt_alpha)
		
		auc_list=find_test_auc(net)
		
		shuff_rocs=find_net_syngo_shuffled_auc(G, opt_alpha, 10)
		#print (net, shuff_rocs)
		rand_rocs=find_net_deg_marched_auc_list(G, opt_alpha, 10)

		plot_test_control_aucs(net, auc_list, shuff_rocs, rand_rocs)

		plot_all_test_aucs(net, auc_list)

	
