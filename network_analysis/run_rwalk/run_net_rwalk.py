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

sys.path.append('../../read_data_functions')
import load_data_functions

sys.path.append('../../ML_functions/')
import find_GO_scores


sys.path.append('../ppi_files/')


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

def find_scores_df(df):
	cols=df.columns
	to_delete=[]
	for item in cols:
		if "Prop" not in item:
			to_delete.append(item)
	final=df.drop(to_delete, axis=1)
	return final

def find_val_df(df, neg_eval):
	pos=df[df['Non-Sample']==1]
	neg=df.loc[neg_eval]
	final=pd.concat([pos, neg])
	return final

def calculate_roc(df, neg_eval):
	final=find_val_df(df, neg_eval)
	#print (final)
	y_test=final['Non-Sample'].tolist()
	final=find_scores_df(final)
	#print (final)
	final['mean']=final.mean(axis=1)
	# calculate the fpr and tpr for all thresholds of the classification
	preds=final['mean'].tolist()
	fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
	roc_auc = metrics.auc(fpr, tpr)
	return fpr, tpr, threshold, roc_auc

def find_prop_scores_df(kernel, nodesets, fraction):
	frames=[]
	for key in list(nodesets.keys()):
		genesets={key: nodesets.get(key)}
		#print ('genesets', genesets)
		genesets_p=net_random_walk_functions.set_p(genesets, fraction)
		#scores=run_propagation(G, genesets, alpha)
		scores= net_random_walk_functions.get_propagated_scores(kernel, genesets, genesets_p, n=1, cores=1, verbose=False)
		#print (scores)
		frames.append(scores)
	df=pd.concat(frames, axis=1)
	#print (df)
	return df

def calc_prop_aucs(df, neg):
	mean_fpr = np.linspace(0, 1, 100)
	tprs = []
	aucs = []
	for j in np.arange(0,14,3):
		#=print ('j', j)
		subdf=df.iloc[:, j:j+3]
		subdf.columns=['Sub-Sample', 'Non-Sample', 'Prop Score']
		fpr, tpr, threshold, roc_auc=calculate_roc(subdf, neg)
		final=pd.DataFrame({'Threshold': threshold, 'TPR': tpr, 'FPR': fpr})
		#print (df)
		#final.to_csv('../propagate_synapse/results/ROC_df_%s.csv'%j)
		#print ('actual', roc_auc)
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.
		aucs.append(roc_auc)
	return mean_fpr, tprs, aucs

def opt_alpha(G, neg):
	alphas=np.arange(0.1, 1, 0.1)

	all_mean_aucs=[]
	for item in alphas:
		kernel=net_random_walk_functions.construct_prop_kernel(G, item, verbose=True)
		df=find_prop_scores_df(kernel, nodesets, 0.8)
		#print (df)
		mean_fpr, tprs, aucs=calc_prop_aucs(df, neg)
		mean_aucs=np.mean(aucs)
		all_mean_aucs.append(mean_aucs)
	return all_mean_aucs

##big_pool=load_data_functions.load_big_pool()
#GO_human=find_GO_scores.find_GO_ont()
#GO_genes=GO_human.genes
#syngo=find_syngo(big_pool, go_genes)
#print (len(seeds)/len(syngo))




filename='../ppi_files/Human_Mentha_converted.csv'
#filename='../ppi_files/BioPlex 3 - HEK293T default edge.csv'
df=make_network_graph_functions.make_mentha_df(filename)
#df=make_network_graph_functions.make_bioplex_df(filename)
#print (df)

G=make_network_graph_functions.make_network_G(df)
#print (G.number_of_edges())
nodes=list(G.nodes())

pos=load_data_functions.get_gene_names('../../run_ML/ML_output/training_genes/updated_positives.csv')
#print (len(pos))
seeds=list(set(pos)&set(nodes))
#print (len(seeds))
nodesets=find_nodesets(G, seeds)
#print (nodesets)
neg=list(set(nodes)-set(seeds))
#all_mean_aucs=opt_alpha(G, neg)
#print (all_mean_aucs)
#print (np.mean(all_mean_aucs))


syngo=load_data_functions.get_gene_names('../../correct_db/corr_syngo_cc.csv')
print (len(syngo))
print (len(seeds)/len(syngo))





