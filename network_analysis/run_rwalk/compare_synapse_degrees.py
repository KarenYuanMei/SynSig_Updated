
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

#import statistics
#from scipy import stats

import run_net_rwalk

import sys

sys.path.append('../../read_data_functions')
import load_data_functions

sys.path.append('../../graph_functions')
import graph_functions



#from propagate_network import calculate_p, fast_random_walk, closed_form_network_propagation, construct_prop_kernel, normalize_network, get_propagated_scores
#from cross_val_ROC import find_scores_df, find_val_df, calculate_roc


def plot_prob_hist(seed_deg, net_deg, net_name, measure_name):
	f = plt.figure()
	bins=np.histogram(np.hstack((seed_deg,net_deg)), bins=40)[1] #get the bin edges

	plt.hist(seed_deg, bins, alpha=0.5, edgecolor='black', color='orange', linewidth=0.5, weights=np.ones_like(seed_deg) / float(len(seed_deg)), bottom=0.00001)
	plt.hist(net_deg, bins, alpha=0.5, edgecolor='black', color='gray', linewidth=0.5, weights=np.ones_like(net_deg) / float((len(net_deg))), bottom=0.00001)

	plt.legend(labels=['Synapse', 'Negatives'])

	plt.axvline(mean(seed_deg), color='orange', linestyle='dashed', linewidth=3)
	min_ylim, max_ylim = plt.ylim()
	plt.text(mean(seed_deg)*1.1, max_ylim*0.9, 'Mean: {:.3f}'.format(mean(seed_deg)))

	
	plt.axvline(mean(net_deg), color='gray', linestyle='dashed', linewidth=3)
	min_ylim, max_ylim = plt.ylim()
	plt.text(mean(net_deg)*1.1, max_ylim*0.9, 'Mean: {:.3f}'.format(mean(net_deg)))

	#plt.ylabel('Non-Synapse Genes in Brain')
	plt.xlim(0, 0.15)
	plt.yscale('log')
	plt.xlabel('%s Centrality'%measure_name, fontweight='bold')
	plt.ylabel('Probability', fontweight = 'bold')
	plt.title('%s Centrality Distributions'%measure_name, fontweight = 'bold')
	plt.grid(b=False)
	#plt.savefig(title, bbox_inches='tight')

	plt.show()
	f.savefig('%s_Net_%s_distr.svg'%(net_name, measure_name), bbox_inches='tight')

def plot_boxplot(seed_deg, net_deg, net_name, measure_name):
	f = plt.figure()
	plt.boxplot([seed_deg, net_deg], showfliers=False)
	#plt.boxplot([x for x in [seed_deg, net_deg]], 0, 'rs', 1)
	#plt.xticks([y+1 for y in range(len([seed_deg, net_deg]))], ['Seed Genes', 'Background Genes'])
	plt.xlabel('Gene Group')
	t = plt.title('%s Centrality of Seed vs. Background Genes'%measure_name)
	plt.xticks([y+1 for y in range(len([seed_deg, net_deg]))], ['Seed Genes', 'Background Genes'])
	#plt.ylim(0,2500)
	plt.yscale('log')
	plt.show()
	f.savefig("%s_Net_seed_%s_centrality_box.svg"%(net_name, measure_name), bbox_inches='tight')
	plt.close()

def plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, net_name, measure_name):
	plt.figure(figsize=(3,6))
	f = plt.figure()
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	#plt.bar(labels, mean_values, yerr=sem, color=['#2d7f5e', '#7f6d5f', '#557f2d','silver'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, yerr=sem, align='center', ecolor='black', capsize=10)

	#plt.ylim(0.5, 1)
	#plt.ylim(1, 10**5)
	#plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	#plt.xticks(rotation=45)
	#plt.figure(figsize=(3,6))
	plt.savefig(net_name+'_'+measure_name+'.svg', format="svg", bbox_inches='tight')


def plot_grouped_bargraph(ref_gene_list, G, net_name):
	seed_genes=list(set(ref_gene_list)&set(G.nodes))
	bg_genes=list(set(G.nodes())-set(seed_genes))

	seed_deg=[G.degree(n) for n in seed_genes]
	bg_deg = [G.degree(n) for n in bg_genes]
	print ('done')

	eig=nx.eigenvector_centrality(G)
	seed_eig=[eig[n] for n in seed_genes]
	bg_eig=[eig[n] for n in bg_genes]
	print ('done')

	f = plt.figure()
	# create data
	seed_means, seed_std = (mean(seed_deg), mean(seed_eig)), (stats.sem(seed_deg), stats.sem(seed_eig))
	bg_means, bg_std = (mean(bg_deg), mean(bg_eig)), (stats.sem(bg_deg), stats.sem(bg_eig))

	ind = np.arange(len(seed_means))  # the x locations for the groups
	width = 0.35  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind - width/2, seed_means, width, yerr=seed_std,
	                label='New')
	rects2 = ax.bar(ind + width/2, bg_means, width, yerr=bg_std,
	                label='Negatives')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Centrality Measures')
	ax.set_yscale('log')
	ax.set_ylim([10**-3, 10**3])
	ax.set_title('Genes')
	ax.set_xticks(ind)
	ax.set_xticklabels(('Degree', 'Eigenvector'))
	ax.legend()
	plt.savefig(net_name+'_new_genes_grouped_bar'+'.svg', format="svg", bbox_inches='tight')


def students_test(seed_deg, net_deg):
	## Cross Checking with the internal scipy function
	t2, p2 = stats.ttest_ind(seed_deg,net_deg)
	print("t = " + str(t2))
	print("p = " + str(p2))
	return t2, p2

def closeness_centrality(G, seed_genes, bg_genes, net_name):
	closeness_d=nx.closeness_centrality(G)
	print ('done')
	seed_closeness=[closeness_d[n] for n in seed_genes]
	print ('done')
	print (seed_closeness)
	print (len(seed_closeness))
	bg_closeness=[closeness_d[n] for n in bg_genes]
	students_test(seed_closeness, bg_closeness)
	plot_prob_hist(seed_closeness, bg_closeness, name_name, 'Closeness')
	plot_boxplot(seed_closeness, bg_closeness, net_name, 'Closeness')

def compare_degrees(G, seed_genes, bg_genes, net_name):
	seed_deg=[G.degree(n) for n in seed_genes]
	net_deg = [G.degree(n) for n in bg_genes]
	print ('done')
	students_test(seed_deg, net_deg)
	plot_prob_hist(seed_deg, net_deg, net_name, 'Degree')
	plot_boxplot(seed_deg, net_deg, net_name, 'Degree')

def eigen_centrality(G, seed_genes, bg_genes, net_name):
	eig=nx.eigenvector_centrality(G)
	seed_eig=[eig[n] for n in seed_genes]
	print ('done')
	print (seed_eig)
	print (len(seed_eig))
	bg_eig=[eig[n] for n in bg_genes]
	students_test(seed_eig, bg_eig)
	plot_prob_hist(seed_eig, bg_eig, net_name, 'Eigenvector')
	plot_boxplot(seed_eig, bg_eig, net_name, 'Eigenvector')

	seed_mean=mean(seed_eig)
	bg_mean=mean(bg_eig)
	mean_values=[seed_mean, bg_mean]
	print ('seed', seed_mean, 'bg', bg_mean)

	seed_sem=stats.sem(seed_eig)
	bg_sem=stats.sem(bg_eig)
	sem=[seed_sem, bg_sem]

	labels=['Synapse', 'Negatives']
	ylabel=['Eigenvector Centrality']
	xlabel=['Genes']
	plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, net_name, 'Eigenvector')

def between_centrality(G, seed_genes, bg_genes, net_name):
	plt.legend(labels=['Seed Genes', 'Background Genes'])
	bet=nx.betweenness_centrality(G)
	seed=[bet[n] for n in seed_genes]
	print ('done')
	print (seed)
	print (len(seed))
	bg=[bet[n] for n in bg_genes]
	students_test(seed, bg)
	plot_prob_hist(seed, bg, net_name, 'Betweenness')
	plot_boxplot(seed, bg, net_name, 'Betweenness')



def find_ntwk_centrality(ref_gene_list, G, net_name):
	#G=make_network_G(network_df)
	
	seed_genes=list(set(ref_gene_list)&set(G.nodes))
	bg_genes=list(set(G.nodes())-set(seed_genes))
	compare_degrees(G, seed_genes, bg_genes, net_name)

	eigen_centrality(G, seed_genes, bg_genes, net_name)
	#closeness_centrality(G, seed_genes, bg_genes, net_name)
	between_centrality(G, seed_genes, bg_genes, net_name)

def plot_degree_bargraph(seed_genes, bg_genes, net_name, measure_name):
	seed_deg=[G.degree(n) for n in seed_genes]
	bg_deg = [G.degree(n) for n in bg_genes]
	mean_values=[mean(seed_deg), mean(bg_deg)]
	sem=[stats.sem(seed_deg), stats.sem(bg_deg)]

	labels=['SynSig (New Genes)', 'Negative Genes']
	xlabel=['Genes']
	ylabel=['%s'%measure_name]
	plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, net_name, measure_name)


if __name__ == '__main__':
	# filename='../Data/Human_Mentha_converted.csv'
	# network_df=make_mentha_df(filename)
	# find_ntwk_centrality(network_df)
	#df=run_net_rwalk.make_bioplex_ppi_df()
	G=run_net_rwalk.df_to_network('mentha')

	syngo_genes=load_data_functions.find_full_syngo()
	
	#filename='../source_data_files/BioPlex 3 - HCT116 default edge.csv'
	#filename='../Data/BioPlex 3 - HEK293T default edge.csv'
	#find_ntwk_centrality(syngo_genes, G, 'mentha')

	synsig_genes=load_data_functions.load_synsig()
	new_genes=list(set(synsig_genes)-set(syngo_genes))
	bg_genes=list(set(G.nodes())-set(synsig_genes)-set(syngo_genes))
	#plot_grouped_bargraph(new_genes, G, 'mentha')
	plot_degree_bargraph(new_genes, bg_genes, 'mentha_new_genes', 'degree')
	



