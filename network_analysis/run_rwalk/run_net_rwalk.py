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

def find_hk_nodes(G):
	hk=pd.read_csv('../../gene_lists/Human_Mouse_Common.csv', sep=';')
	print (hk)
	hk=hk['Human'].tolist()
	nodes=list(G.nodes())
	hk_nodes=list(set(nodes)&set(hk))
	return hk_nodes

def find_net_syngo_test_auc(G,opt_alpha):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	syngo_nodes=find_syngo_nodes(G)
	print ('syngo nodes', len(syngo_nodes))
	#syngo_nodes=find_hk_nodes(G)

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

def find_deg_matched_auc(G, opt_alpha, kernel, buckets):
	nodes=list(G.nodes())
	cv_seeds=find_cv_seeds(nodes)
	

	rand_seeds=net_random_walk_functions.find_rand_samples(G, cv_seeds, buckets)
	print ('rand', rand_seeds)

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

def calc_plot_opt_alpha(G, cv_seedsets, neg, net):
	alpha_cvs, all_mean_aucs=sweep_alpha_aucs(G, cv_seedsets, neg)
	alpha_df=make_sweep_alpha_df(alpha_cvs, all_mean_aucs)
	alpha_df.to_csv('%s_alpha_df.csv'%net)
	alpha_df=pd.read_csv('%s_alpha_df.csv'%net, index_col=[0])
	graph_functions.plot_alpha(alpha_df, net)
	print ('done')
	opt_alpha=find_opt_alpha(all_mean_aucs)
	return alpha_cvs, opt_alpha

def calc_auc_bootstrap(set1_predictions):
	#set1_predictions and set2_predictions should be the output from find_true_y
	#returns a confidence interval for the difference between the auc scores for the two sets
	scores = set1_predictions['avg_scores']

	set1_labels = set1_predictions['label']

	num_bootstrap_samples = 10000

	bootstrapped_auc = []
	for i in range(num_bootstrap_samples):
		#indices = random.randint(0,len(scores))
		indices=list(np.random.randint(low = 0,high=len(scores),size=len(scores)))
		#print (indices)

		set1_auc = roc_auc_score(set1_labels[indices],scores[indices])

		bootstrapped_auc_diffs.append(set1_auc) 


	conf_interval_sizes = [0.95,0.99, 0.999]
	conf_intervals = {}

	bootstrapped_auc.sort()

	for interval_size in conf_interval_sizes:
		#print (interval_size)
		lower_bound_index = int(num_bootstrap_samples*((1-interval_size)/2))
		#print (lower_bound_index)

		lower_bound = bootstrapped_auc[lower_bound_index]

		upper_bound_index = int(num_bootstrap_samples*(interval_size+((1-interval_size)/2)))
		#print (upper_bound_index)
		upper_bound = bootstrapped_auc[upper_bound_index]

		conf_intervals[interval_size] = (lower_bound,upper_bound)

	return conf_intervals


if __name__ == '__main__':

	net_names=['mentha', 'bioplex']
	for net in net_names:

		G=df_to_network(net)

		nodes=list(G.nodes())

		cv_seeds=find_cv_seeds(nodes)

		cv_seedsets=find_cv_nodesets(G, cv_seeds)
		#print (cv_seedsets)

		neg=list(set(nodes)-set(cv_seeds))

		#alpha_cvs, opt_alpha=calc_plot_opt_alpha(G, cv_seedsets, neg, net)

		opt_alpha=0.5

		print ('opt_alpha', opt_alpha)

		#tprs, mean_fpr, aucs=alpha_cvs[opt_alpha]
		#tprs, mean_fpr, aucs=find_single_alpha_auc(G, cv_seedsets, opt_alpha, neg)
		#print (net, aucs) #0.6708522690436207
		
		fpr, tpr, threshold, roc_auc=find_net_syngo_test_auc(G, opt_alpha)
		print (net, 'single threshold', roc_auc)
		#graph_functions.plot_single_ROC(tpr, fpr, roc_auc, '%s_test'%net)

		
		shuff_rocs=find_net_syngo_shuffled_auc(G, opt_alpha)
		print (net, shuff_rocs)
	
		kernel=net_random_walk_functions.construct_prop_kernel(G, opt_alpha, verbose=True)
		bg=list(set(nodes)-set(cv_seeds))
		buckets=net_random_walk_functions.make_seed_bg_buckets(G, cv_seeds, bg)
		print ('buckets')
		all_rand_rocs=[]
		for i in range(10):
			rand_seed_rocs=find_deg_matched_auc(G, opt_alpha, kernel, buckets)
			#print (rand_seed_rocs)
			all_rand_rocs.append(rand_seed_rocs)
		print (net, all_rand_rocs)

		#control_df=pd.DataFrame({'shuff': all_shuff_rocs, 'rand_seed': all_rand_rocs})
		#control_df.to_csv('%s_control.csv'%net)

		#mentha:
		#shuff net:  [0.6427267889496402, 0.6303507526964867, 0.6357507069450844, 0.6435493207635116, 0.6303259841115905, 0.6363102487437159, 0.6317246372375604, 0.6395814471619632, 0.6357818522659295, 0.638829126567066])
		#degree matched: 'mentha', [0.6326565479466931, 0.6250954272000686, 0.6255695090503928, 0.6272859934267048, 0.6310719489691188, 0.6246677285667572, 0.6304887788488471, 0.6288220511562175, 0.6352668877022264, 0.6276963247716165]

		#bioplex:
		#degree matched: 'bioplex', 'bioplex', [0.5461990699275066, 0.5657750669501984, 0.5582950107790665, 0.5438948986230199, 0.5429975923521106, 0.5761597777653801, 0.5619553439315107, 0.5339053373455318, 0.5525482462398235, 0.561227381829396])


		
