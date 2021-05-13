
#Goal: determine if synapse genes have more interactors than background genes

#things to do: still need to calculate the probability signifiance between the two distributions

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import random

import scipy.stats as stats
import sklearn.metrics as metrics
from statistics import mean

import run_net_rwalk

sys.path.append('../../read_data_functions')
import load_data_functions

sys.path.append('../../graph_functions')
import graph_functions



#from propagate_network import calculate_p, fast_random_walk, closed_form_network_propagation, construct_prop_kernel, normalize_network, get_propagated_scores
#from cross_val_ROC import find_scores_df, find_val_df, calculate_roc


def plot_prob_hist(seed_deg, net_deg, name):
	f = plt.figure()
	bins=np.histogram(np.hstack((seed_deg,net_deg)), bins=40)[1] #get the bin edges

	plt.hist(seed_deg, bins, alpha=0.5, edgecolor='black', linewidth=0.5, weights=np.ones_like(seed_deg) / float(len(seed_deg)))
	plt.hist(net_deg, bins, alpha=0.5, edgecolor='black', linewidth=0.5, weights=np.ones_like(net_deg) / float((len(net_deg))))

	#plt.ylabel('Non-Synapse Genes in Brain')
	plt.xlabel('%s Centrality'%name, fontweight='bold')
	plt.ylabel('Probability', fontweight = 'bold')
	plt.title('%s Centrality Distributions'%name, fontweight = 'bold')
	plt.grid(b=False)
	#plt.savefig(title, bbox_inches='tight')
	plt.show()
	f.savefig('Net_%s_distr.svg'%name, bbox_inches='tight')

def plot_boxplot(seed_deg, net_deg, name):
	f = plt.figure()
	plt.boxplot([seed_deg, net_deg], showfliers=False)
	#plt.boxplot([x for x in [seed_deg, net_deg]], 0, 'rs', 1)
	#plt.xticks([y+1 for y in range(len([seed_deg, net_deg]))], ['Seed Genes', 'Background Genes'])
	plt.xlabel('Gene Group')
	t = plt.title('%s Centrality of Seed vs. Background Genes'%name)
	plt.xticks([y+1 for y in range(len([seed_deg, net_deg]))], ['Seed Genes', 'Background Genes'])
	#plt.ylim(0,2500)
	plt.yscale('log')
	plt.show()
	f.savefig("Net_seed_%s_centrality_box.svg"%name, bbox_inches='tight')

def students_test(seed_deg, net_deg):
	## Cross Checking with the internal scipy function
	t2, p2 = stats.ttest_ind(seed_deg,net_deg)
	print("t = " + str(t2))
	print("p = " + str(p2))
	return t2, p2

def closeness_centrality(G, seed_genes, bg_genes):
	closeness_d=nx.closeness_centrality(G)
	print ('done')
	seed_closeness=[closeness_d[n] for n in seed_genes]
	print ('done')
	print (seed_closeness)
	print (len(seed_closeness))
	bg_closeness=[closeness_d[n] for n in bg_genes]
	students_test(seed_closeness, bg_closeness)
	plot_prob_hist(seed_closeness, bg_closeness, 'Closeness')
	plot_boxplot(seed_closeness, bg_closeness, 'Closeness')

def compare_degrees(G, seed_genes, bg_genes):
	seed_deg=[G.degree(n) for n in seed_genes]
	net_deg = [G.degree(n) for n in bg_genes]
	print ('done')
	students_test(seed_deg, net_deg)
	plot_prob_hist(seed_deg, net_deg, 'Degree')
	plot_boxplot(seed_deg, net_deg, 'Degree')

def eigen_centrality(G, seed_genes, bg_genes):
	eig=nx.eigenvector_centrality(G)
	seed_eig=[eig[n] for n in seed_genes]
	print ('done')
	print (seed_eig)
	print (len(seed_eig))
	bg_eig=[eig[n] for n in bg_genes]
	students_test(seed_eig, bg_eig)
	plot_prob_hist(seed_eig, bg_eig, 'Eigenvector')
	plot_boxplot(seed_eig, bg_eig, 'Eigenvector')

def between_centrality(G, seed_genes, bg_genes):
	plt.legend(labels=['Seed Genes', 'Background Genes'])
	bet=nx.betweenness_centrality(G)
	seed=[bet[n] for n in seed_genes]
	print ('done')
	print (seed)
	print (len(seed))
	bg=[bet[n] for n in bg_genes]
	students_test(seed, bg)
	plot_prob_hist(seed, bg, 'Betweenness')
	plot_boxplot(seed, bg, 'Betweenness')


def find_ntwk_centrality(G):
	#G=make_network_G(network_df)
	syngo_genes=load_data_functions.find_full_syngo()
	seed_genes=list(set(syngo_genes)&set(G.nodes))
	bg_genes=list(set(G.nodes())-set(seed_genes))
	compare_degrees(G, seed_genes, bg_genes)

	eigen_centrality(G, seed_genes, bg_genes)
	closeness_centrality(G, seed_genes, bg_genes)
	between_centrality(G, seed_genes, bg_genes)


if __name__ == '__main__':
	# filename='../Data/Human_Mentha_converted.csv'
	# network_df=make_mentha_df(filename)
	# find_ntwk_centrality(network_df)
	df=run_net_rwalk.make_bioplex_ppi_df()
	G=run_net_rwalk.df_to_network(net)
	
	
	#filename='../source_data_files/BioPlex 3 - HCT116 default edge.csv'
	#filename='../Data/BioPlex 3 - HEK293T default edge.csv'
	find_ntwk_centrality(G)







