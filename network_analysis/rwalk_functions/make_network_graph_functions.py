#Goal: functions that convert a ppi file into a dataframe and then into an undirected network graph

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append('../../read_data_functions/')
import load_data_functions

def clean_names(gene_list):
	entries=[]
	for item in gene_list:
		if '{' in item:
			entry=item[:item.index('{')-1]
			#print (item, entry)
		else:
			entry=item
		entries.append(entry)
	return entries

def make_mentha_df(filename):
	df=pd.read_csv(filename)
	node1=df['mapped Gene A'].tolist()
	node2=df['mapped Gene B'].tolist()
	node1=clean_names(node1)
	node2=clean_names(node2)
	new_df=pd.DataFrame({'Node1':node1, 'Node2': node2})
	return new_df

def make_bioplex_df(filename):
	df=pd.read_csv(filename)
	names=df['name'].tolist()
	
	node1_list=[]
	for item in names:
		node1=item[:item.find(' (interacts')]
		node1_list.append(node1)

	node2_list=[]
	for item in names:
		node2=item[item.find('-with) ')+7:]
		node2_list.append(node2)

	df=pd.DataFrame({'Node1': node1_list, 'Node2': node2_list})
	return df

#load the converted the network file into an undirected network graph
def make_network_G(new):
	graph= nx.from_pandas_edgelist(new, 'Node1', 'Node2')
	G=graph.to_undirected()
	return G

def find_hek_genes():
	exp_df=pd.read_csv('../../source_data_files/expression_files/rna_celline.tsv', sep='\t')
	print (exp_df)
	hek=exp_df[exp_df['Cell line']=='HEK 293']
	print (hek)
	hek_nonzero=hek[hek['TPM']>0]
	print (hek_nonzero)
	hek_genes=hek_nonzero['Gene name'].tolist()
	df=pd.DataFrame({'genes': hek_genes})
	df.to_csv('hek_genes.csv')
	return hek_genes

def filter_by_hek_genes(G, hek_genes):
	network_nodes=list(G.nodes())
	#hek_genes_in_network=list(set(hek_genes)&set(network_nodes))
	non_hek=list(set(network_nodes)-set(hek_genes))
	G.remove_nodes_from(non_hek)
	return G







