#Goal: use the environment jh_network to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import time
#import matplotlib.pyplot as plt
import sys
sys.path.append('../rwalk_functions/')
import make_network_graph_functions
import net_random_walk_functions

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


kernel=net_random_walk_functions.construct_prop_kernel(G, 0.4, verbose=True)
print (kernel)

frames=[]
for key in list(nodesets.keys()):
	genesets={key: nodesets.get(key)}
	#print ('genesets', genesets)
	genesets_p=net_random_walk_functions.set_p(genesets)
	#scores=run_propagation(G, genesets, alpha)
	scores= net_random_walk_functions.get_propagated_scores(kernel, genesets, genesets_p, n=1, cores=1, verbose=False)
	#print (scores)
	frames.append(scores)
df=pd.concat(frames, axis=1)
print (df)

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
for j in np.arange(0,14,3):
	print ('j', j)
	subdf=df.iloc[:, j:j+3]
	subdf.columns=['Sub-Sample', 'Non-Sample', 'Prop Score']
	fpr, tpr, threshold, roc_auc=calculate_roc(subdf, neg_eval)
	final=pd.DataFrame({'Threshold': threshold, 'TPR': tpr, 'FPR': fpr})
	print (df)
	#final.to_csv('../propagate_synapse/results/ROC_df_%s.csv'%j)
	print ('actual', roc_auc)
	tprs.append(np.interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.
	aucs.append(roc_auc)

print (aucs)