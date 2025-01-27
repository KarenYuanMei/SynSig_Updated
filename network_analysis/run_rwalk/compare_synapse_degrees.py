
#Goal: determine if synapse genes have more interactors than background genes

#things to do: still need to calculate the probability signifiance between the two distributions

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
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


import itertools

#from propagate_network import calculate_p, fast_random_walk, closed_form_network_propagation, construct_prop_kernel, normalize_network, get_propagated_scores
#from cross_val_ROC import find_scores_df, find_val_df, calculate_roc


def plot_prob_hist(seed_deg, net_deg, net_name, measure_name):
	f = plt.figure()
	bins=np.histogram(np.hstack((seed_deg,net_deg)), bins=40)[1] #get the bin edges

	plt.hist(seed_deg, bins, alpha=0.5, edgecolor='black', color='rebeccapurple', linewidth=0.5, weights=np.ones_like(seed_deg) / float(len(seed_deg)), bottom=0.00005)
	plt.hist(net_deg, bins, alpha=0.5, edgecolor='black', color='gray', linewidth=0.5, weights=np.ones_like(net_deg) / float((len(net_deg))), bottom=0.00005)

	#plt.hist(seed_deg, bins, alpha=0.5, edgecolor='black', color='rebeccapurple', linewidth=0.5)
	#plt.hist(net_deg, bins, alpha=0.5, edgecolor='black', color='gray', linewidth=0.5)

	plt.legend(labels=['Synapse', 'Negatives'])

	#plt.axvline(mean(seed_deg), color='orange', linestyle='dashed', linewidth=3)
	#min_ylim, max_ylim = plt.ylim()
	#plt.text(mean(seed_deg)*1.1, max_ylim*0.9, 'Mean: {:.3f}'.format(mean(seed_deg)))

	
	#plt.axvline(mean(net_deg), color='gray', linestyle='dashed', linewidth=3)
	#min_ylim, max_ylim = plt.ylim()
	#plt.text(mean(net_deg)*1.1, max_ylim*0.9, 'Mean: {:.3f}'.format(mean(net_deg)))

	#plt.ylabel('Non-Synapse Genes in Brain')
	plt.xlim(0, 5)
	#plt.yscale('log')
	plt.xlabel('%s Centrality'%measure_name, fontweight='bold')
	plt.ylabel('Probability', fontweight = 'bold')
	plt.title('%s Centrality Distributions'%measure_name, fontweight = 'bold')
	plt.grid(b=False)
	#plt.savefig(title, bbox_inches='tight')

	#plt.show()
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
	plt.bar(labels, mean_values, yerr=sem, align='center', ecolor='black', capsize=10, bottom=10**-4)

	plt.ylim(10**-4, 10**-1)
	#plt.ylim(1, 10**5)
	plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	#plt.xticks(rotation=45)
	#plt.figure(figsize=(3,6))
	plt.savefig(net_name+'_'+measure_name+'.svg', format="svg", bbox_inches='tight')


def plot_grouped_mentha_bioplex_bargraph(ref_gene_list, G1,net1_name, G2, net2_name):
	seed_genes=list(set(ref_gene_list)&set(G1.nodes))
	bg_genes=list(set(G1.nodes())-set(seed_genes))

	seed_deg_1=[G1.degree(n) for n in seed_genes]
	bg_deg_1= [G1.degree(n) for n in bg_genes]
	print ('done')

	seed_genes=list(set(ref_gene_list)&set(G2.nodes))
	bg_genes=list(set(G2.nodes())-set(seed_genes))

	seed_deg_2=[G2.degree(n) for n in seed_genes]
	bg_deg_2= [G2.degree(n) for n in bg_genes]
	print ('done')


	f = plt.figure()
	# create data
	seed_means, seed_std = (mean(seed_deg_1), mean(seed_deg_2)), (stats.sem(seed_deg_1), stats.sem(seed_deg_2))
	bg_means, bg_std = (mean(bg_deg_1), mean(bg_deg_2)), (stats.sem(bg_deg_1), stats.sem(bg_deg_2))

	ind = np.arange(len(seed_means))  # the x locations for the groups
	width = 0.35  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind - width/2, seed_means, width, yerr=seed_std,
	                label='New')
	rects2 = ax.bar(ind + width/2, bg_means, width, yerr=bg_std,
	                label='Negatives')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Number of Protein Partners')
	#ax.set_yscale('log')
	#ax.set_ylim([10**-3, 10**3])
	ax.set_title('Genes')
	ax.set_xticks(ind)
	ax.set_xticklabels(('Compiled', 'HEK293T'))
	ax.legend()
	plt.savefig(net1_name+net2_name+'_new_genes_grouped_bar'+'.svg', format="svg", bbox_inches='tight')


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
	#print (seed_deg)
	bg_deg = [G.degree(n) for n in bg_genes]
	#print (bg_deg)
	mean_values=[mean(seed_deg), mean(bg_deg)]
	print (mean_values)
	sem=[stats.sem(seed_deg), stats.sem(bg_deg)]
	print (sem)

	labels=['SynSig (New Genes)', 'Negative Genes']
	xlabel=['Genes']
	ylabel=['%s'%measure_name]
	#plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, net_name, measure_name)
	plt.figure(figsize=(3,6))
	f = plt.figure()
	x_pos=np.arange(len(labels))
	#plt.bar(labels, mean_values, yerr=sem, color=['#7f6d5f', '#2d7f5e', '#557f2d','silver', 'dimgray', 'rosybrown'], align='center', ecolor='black', capsize=10)
	#plt.bar(labels, mean_values, yerr=sem, color=['#2d7f5e', '#7f6d5f', '#557f2d','silver'], align='center', ecolor='black', capsize=10)
	plt.bar(labels, mean_values, yerr=sem, align='center', ecolor='black', capsize=10, bottom=10**-4)

	#plt.ylim(10**-4, 10**-1)
	#plt.ylim(1, 10**5)
	#plt.yscale('log')
	# Create legend & Show graphic
	#plt.legend()
	plt.xlabel(xlabel, fontweight='bold')
	plt.ylabel(ylabel, fontweight='bold')
	#plt.xticks(rotation=45)
	#plt.figure(figsize=(3,6))
	plt.savefig(net_name+'_'+measure_name+'.svg', format="svg", bbox_inches='tight')

def plot_eigen_bargraph(G, seed_genes, bg_genes, net_name, measure_name):
	eig=nx.eigenvector_centrality(G)
	seed_eig=[eig[n] for n in seed_genes]
	bg_eig=[eig[n] for n in bg_genes]

	mean_values=[mean(seed_eig), mean(bg_eig)]
	print (mean_values)
	sem=[stats.sem(seed_eig), stats.sem(bg_eig)]
	print (sem)

	labels=['SynSig (New Genes)', 'Negative Genes']
	xlabel=['Genes']
	ylabel=['%s'%measure_name]
	plot_bargraph_with_errorbar(labels, mean_values, sem, xlabel, ylabel, net_name, measure_name)

def find_shorted_path_nodeset(node_pairs, graph):

	paths=[]
	for pair in node_pairs:
		path=nx.shortest_path_length(graph, source=pair[0], target=pair[1])
		#print (pair[0], pair[1])
		#print (path)
		paths.append(path)
	return paths

def find_shortested_paths_all_components(S, genelist):
	all_paths=[]
	for graph in S:
		nodes=list(graph.nodes)
		overlap=list(set(nodes)&set(genelist))
		overlap_pairs=list(itertools.combinations(overlap, 2))

		paths=find_shorted_path_nodeset(overlap_pairs, graph)
		all_paths.append(paths)

	flat_paths=[item for sublist in all_paths for item in sublist]
	print (flat_paths[:5])
	return flat_paths

def find_shortest_path_syn_target(path_dict, target_genes):
			
	new_dict = { your_key: path_dict[your_key] for your_key in target_genes}
	#sub_paths=new_dict.items()
	#print (sub_paths)
	#print (len(new_dict.keys()))
	values=new_dict.values()
	avg_paths=mean(values)
	#print (avg_paths)
	return avg_paths

if __name__ == '__main__':

	
	# # filename='../Data/Human_Mentha_converted.csv'
	# # network_df=make_mentha_df(filename)
	# # find_ntwk_centrality(network_df)
	# #df=run_net_rwalk.make_bioplex_ppi_df()
	# G=run_net_rwalk.df_to_network('mentha')

	
	# # # #G2=run_net_rwalk.df_to_network('bioplex')


	# # # #syngo_genes=load_data_functions.find_full_syngo()
	
	# # # #filename='../source_data_files/BioPlex 3 - HCT116 default edge.csv'
	# # # #filename='../Data/BioPlex 3 - HEK293T default edge.csv'
	# # # #find_ntwk_centrality(syngo_genes, G, 'mentha')

	# synsig_genes=load_data_functions.load_synsig()

	# bg_genes=list(set(G.nodes())-set(synsig_genes))

	
	# connected_comp=list(nx.connected_components(G))
	# # print (len(connected_comp))

	# S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
	# print (len(S))
	
	# all_syn_paths=[]
	# all_bg_paths=[]
	# for syn_gene in synsig_genes:

	# 	syn_paths=[]
	# 	bg_paths=[]

	# 	for graph in S:

	# 		if syn_gene in graph.nodes:

	# 			syn_subset=list(set(graph.nodes)&set(synsig_genes))
	# 			paths=nx.single_source_shortest_path_length(graph, syn_gene)
	# 			avg_syn_paths=find_shortest_path_syn_target(paths, syn_subset)

	# 			bg_subset=list(set(graph.nodes)&set(bg_genes))

	# 			avg_bg_paths=find_shortest_path_syn_target(paths, bg_subset)

	# 		syn_paths.append(avg_syn_paths)
	# 		bg_paths.append(avg_bg_paths)

	# 	all_syn_paths.append(mean(syn_paths))
	# 	all_bg_paths.append(mean(bg_paths))


	# #flat_syn_paths=[item for sublist in all_syn_paths for item in sublist]
	# print (len(all_syn_paths))
	# print (mean(all_syn_paths))

	# #flat_bg_paths=[item for sublist in all_bg_paths for item in sublist]
	# print (len(all_bg_paths))
	# print (mean(all_bg_paths))


	# synsig_path_df=pd.DataFrame({'paths': all_syn_paths})
	# synsig_path_df.to_csv('synsig_path.csv')

	

	# all_paths=[]
	# for graph in S:
	# 	syn_subset=list(set(graph.nodes)&set(synsig_genes))
	# 	bg_subset=list(set(graph.nodes)&set(bg_genes))
	# 	syn_bg_pairs = list(itertools.product(syn_subset, bg_subset))

	# 	paths=find_shorted_path_nodeset(syn_bg_pairs, graph)
	# 	all_paths.append(paths)

	# flat_paths=[item for sublist in all_paths for item in sublist]
	# print (flat_paths[:5])
	# print (mean(flat_paths))



	synsig_paths=pd.read_csv('synsig_path.csv')
	
	synsig_paths=synsig_paths['paths'].tolist()
	# #print (synsig_paths)

	synsig_bg_paths=pd.read_csv('synsig_bg_path.csv')
	synsig_bg_paths=synsig_bg_paths['paths'].tolist()
	# #print (synsig_bg_paths)

	plot_prob_hist(synsig_paths, synsig_bg_paths, 'mentha', 'paths')
	# # #plot_eigen_bargraph(G, synsig_genes, bg_genes, 'mentha_synsig', 'eigen')
	# # #eigen_centrality(G, new_genes, bg_genes, 'mentha_new_genes')

	# # #plot_grouped_mentha_bioplex_bargraph(synsig_genes, G1,'mentha', G2, 'bioplex')

	# # bins = np.linspace(0, 10, 100)

	# import collections
	# synsig_counter = collections.Counter(synsig_paths)
	# total_synsig=len(synsig_paths)
	# synsig_bg_counter=collections.Counter(synsig_bg_paths)
	# total_synsig_bg=len(synsig_bg_paths)

	# print (synsig_counter)
	# print (synsig_bg_counter)

	# count = synsig_counter.items()

	# synsig_percentages = {x: int(float(y) / len(synsig_paths)*100) for x, y in count}

	# count = synsig_bg_counter.items()

	
	
	# synsig_bg_percentages = {x: int(float(y) / len(synsig_bg_paths)*100) for x, y in count}
	# print (synsig_percentages)

	# synsig_bg_percentages.pop(6, None)
	# synsig_bg_percentages.pop(7, None)
	# print (synsig_bg_percentages)

	# names = list(synsig_percentages.keys())
	# values = list(synsig_percentages.values())

	# names2 = list(synsig_bg_percentages.keys())
	# values2 = list(synsig_bg_percentages.values())


	

	# # create data
	# x = np.arange(5)

	# width = 0.40
	  
	# # plot data in grouped manner of bar type
	# plt.bar(x-0.2, values, width, color='rebeccapurple')
	# plt.bar(x+0.2, values2, width, color='gray')
	# plt.show()
