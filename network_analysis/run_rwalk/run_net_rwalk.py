#Goal: use the environment jh_network to run random walk
# network propagation on bioplex and mentha

import scipy as sp
import networkx as nx
import pandas as pd

from scipy import interp

import pathlib

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



def make_bioplex_ppi_df():
	filename=load_data_functions.load_bioplex_file()
	df=make_network_graph_functions.make_bioplex_df(filename)
	return df

def make_filtered_bioplex():
	bioplex=make_bioplex_ppi_df()
	G=make_network_graph_functions.make_network_G(bioplex)
	hek_genes=load_data_functions.load_hek_genes()
	#print (hek_genes)
	bio_fil=make_network_graph_functions.filter_by_hek_genes(G, hek_genes)
	return bio_fil


def make_mentha_ppi_df():
	#filename='../ppi_files/Human_Mentha_converted.csv'
	filename=load_data_functions.load_mentha_file()
	df=make_network_graph_functions.make_mentha_df(filename)
	return df

def find_cv_seeds(nodes):
	#find the overlap between positive synapse genes (training) and the nodes in the network
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
	#find the synapse genes that are not in the training set
	non_seed_pos=list(set(syngo_nodes)-set(cv_seeds))

	#make ordered list of training genes first and then non-training positives
	ordered_test=cv_seeds+non_seed_pos

	#make a set with key 'syngo' and the values are the ordered list of training genes and non-training positives
	ordered_set={'syngo': ordered_test}

	#find what fraction of the ordered list is training genes
	fraction=len(cv_seeds)/float(len(ordered_test))
	return ordered_set, fraction

def find_syngo_nodes(G):
	syngo=load_data_functions.get_gene_names('../../source_data_files/correct_db/corr_syngo_cc.csv')
	nodes=list(G.nodes())
	syngo_nodes=list(set(nodes)&set(syngo))
	#print ('syngo nodes', len(syngo_nodes))
	return syngo_nodes

def find_synapse_nodes(G):
	syngo=load_data_functions.get_gene_names('../../source_data_files/correct_db/corr_syngo_cc.csv')
	syndb=load_data_functions.get_gene_names('../../source_data_files/correct_db/SynDB.csv')
	synsysnet=load_data_functions.get_gene_names('../../source_data_files/correct_db/synsysnet.csv')

	synapse=list(set(syngo)&set(syndb)&set(synsysnet))

	nodes=list(G.nodes())
	synapse_nodes=list(set(nodes)&set(synapse))
	return synapse_nodes

def find_hk_nodes(G):
	hk=pd.read_csv('../../source_data_files/gene_lists/Human_Mouse_Common.csv', sep=';')
	#print (hk)
	hk=hk['Human'].tolist()
	nodes=list(G.nodes())
	hk_nodes=list(set(nodes)&set(hk))
	return hk_nodes

def find_net_test_auc(G,opt_alpha, gold_standards):
	nodes=list(G.nodes())

	#find the positive training synapse genes (same as in random forest) in the network: cv_seeds
	cv_seeds=find_cv_seeds(nodes)

	if gold_standards=='syngo':
		pos_nodes=find_syngo_nodes(G)
		print ('syngo nodes', len(pos_nodes))
	elif gold_standards=='hk':
		pos_nodes=find_hk_nodes(G)
	elif gold_standards=='synapse':
		pos_nodes=find_synapse_nodes(G)

	#syngo_nodes=find_hk_nodes(G)

	#make ordered set: training genes + pos_nodes (excludes the training genes)
	ordered_set, seed_fraction=find_ordered_set(pos_nodes, cv_seeds)

	neg=list(set(nodes)-set(pos_nodes))

	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)

	#seed network propagation with the syngo training genes (same as in random forest), then evaluate how well it recovers the gold standards
	df=net_random_walk_functions.find_prop_scores_df(kernel, ordered_set, seed_fraction)
	fpr, tpr, threshold, roc_auc=net_roc_functions.calc_net_test_roc(df, neg)
	return fpr, tpr, threshold, roc_auc

def find_net_shuffled_auc(G, opt_alpha, ref_list_name, iterations):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)

	if ref_list_name == 'syngo':
		ref_nodes=find_syngo_nodes(G)

	elif ref_list_name == 'synapse':
		ref_nodes=find_synapse_nodes(G)

	ordered_set, seed_fraction=find_ordered_set(ref_nodes, cv_seeds)

	neg=list(set(nodes)-set(ref_nodes))

	shuff_scores, tprs, mean_fpr=net_roc_functions.find_shuff_aucs(G, ordered_set, neg, opt_alpha, seed_fraction, iterations)
	#print (shuff_rocs)
	return shuff_scores, tprs, mean_fpr

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
	return fpr, tpr, threshold, roc_auc

def df_to_network(name):
	if name == 'mentha':
		net_df=make_mentha_ppi_df()
		G=make_network_graph_functions.make_network_G(net_df)
		print ('orig', len(list(G.nodes())))

	if name == 'bioplex':
		net_df=make_bioplex_ppi_df()
		hek_genes=load_data_functions.load_hek_genes()
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

def find_test_auc_list(net):
	genelists=['syngo', 'hk', 'synapse']

	auc_list=[]
	for item in genelists:
		fpr, tpr, threshold, roc_auc=find_net_test_auc(G, opt_alpha, item)
		#graph_functions.plot_single_ROC(tpr, fpr, roc_auc, '%s_%s_test'%(net, item))
		auc_list.append(roc_auc)
	return auc_list

def find_test_auc(G, opt_alpha, ref_list):
	#ref_list is the list of gold standards: can be syngo, hk, or synapse
	fpr, tpr, threshold, roc_auc=find_net_test_auc(G, opt_alpha, ref_list)
	return fpr, tpr, threshold, roc_auc

def find_net_deg_marched_auc_list(G, opt_alpha, iterations):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	bg=list(set(nodes)-set(cv_seeds))
	buckets=net_random_walk_functions.make_seed_bg_buckets(G, cv_seeds, bg)
	print ('newbuckets', buckets)
	kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)

	all_rand_rocs=[]
	tprs=[]
	mean_fpr=np.linspace(0, 1, 100)
	for i in range(iterations):
		fpr, tpr, threshold, roc_auc=find_deg_matched_auc(G, opt_alpha, kernel, buckets)
		#print (rand_seed_rocs)
		all_rand_rocs.append(roc_auc)
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0]=0
	print (all_rand_rocs)
	return all_rand_rocs, tprs, mean_fpr

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

	graph_functions.plot_bargraph(labels, auc_list, xlabel, ylabel, net)

if __name__ == '__main__':

	#net_names=['mentha', 'bioplex']
	net_names=['mentha']
	for net in net_names:

		G=df_to_network(net)

		#alpha_cvs, opt_alpha=calc_plot_opt_alpha(G, net)

		#print ('opt_alpha', opt_alpha)
		
		#auc_list=find_test_auc(net)
		#print (auc_list)
		opt_alpha=0.4

		fpr, tpr, threshold, roc_auc=find_test_auc(G, opt_alpha, 'syngo')
		graph_functions.plot_single_ROC(tpr, fpr, roc_auc, 'purple', net)
		print (roc_auc)

		shuff_rocs, tprs, mean_fpr=find_net_shuffled_auc(G, opt_alpha, 'syngo', 10)
		print (net, shuff_rocs)

		# shuff_rocs, tprs, mean_fpr=find_net_shuffled_auc(G, opt_alpha, 'synapse', 10)
		# print (net, shuff_rocs)

		mean_auc=np.mean(shuff_rocs)
		mean_tpr = np.mean(tprs, axis=0)

		name=net+'shuffled_mean'

		graph_functions.plot_single_ROC(mean_tpr, mean_fpr, mean_auc, 'blue', name )
		#plt.savefig('%s_shuffled_mean_ROC.svg'%net, format="svg")


		all_rand_rocs, tprs, mean_fpr=find_net_deg_marched_auc_list(G, opt_alpha, 10)
		print (net, all_rand_rocs)
		mean_auc=np.mean(all_rand_rocs)
		mean_tpr = np.mean(tprs, axis=0)

		name=net+'random_seed_mean'
		graph_functions.plot_single_ROC(mean_tpr, mean_fpr, mean_auc,'gray', name)
		

		# plot_test_control_aucs(net, auc_list, shuff_rocs, rand_rocs)

		# plot_all_test_aucs(net, auc_list)

		# graph_functions.plot_single_ROC(tpr, fpr, auc, name)
	

#mentha: 'syngo, hk, synapse': [0.7456864401864853, 0.7196956308940988, 0.815815521185644]
#shuffled net: evluating with syngo:
#mentha: ('mentha', [0.6329138649302612, 0.6384526977754723, 0.6341973340679333, 0.6368434781161163, 0.6323335819589118, 0.6295272811531072, 0.6411557625824411, 0.6228213042586394, 0.6373723444586566, 0.6334948862605102])
#degree-matched random seeds: [0.621572051208755, 0.623082056892779, 0.6253766278742644, 0.6261662178121625, 0.6182276961198311, 0.6272120776766205, 0.6195359610255194, 0.6219735479172066, 0.6167694062470768, 0.6308035741716488]

#bioplex: 'syngo, hk, synapse': [0.6708521256093805, 0.5396605856727543, 0.7187177546631718]
#'bioplex shuffled net', [0.5617163111696832, 0.5735153552091588, 0.5615875072219141, 0.5679666016240199, 0.5768385831221503, 0.5596699348636428, 0.5726495861348431, 0.5705051008084527, 0.5633273645564812, 0.5685205446599719]
#degree-matched: [0.5492561747815808, 0.5553582929022335, 0.5558399455537568, 0.5610399615586059, 0.5583944676268677, 0.5607942381670058, 0.5623108482523445, 0.5533689814122542, 0.5519377886805794, 0.559837889818487]