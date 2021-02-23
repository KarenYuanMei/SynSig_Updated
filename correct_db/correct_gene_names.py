import pandas as pd
import networkx as nx
import numpy as np

from mlxtend.evaluate import permutation_test
from scipy import stats

import ddot
from ddot import Ontology
import csv

def load_syngo_genes():
	#syngo=Ontology.from_table('/Users/karenmei/Documents/SynSig_August2020/SynSig/prev_databases/SynGO_BP.txt')
	#syngo_bp_genes=syngo.genes
	syngo=Ontology.from_table('/Users/karenmei/Documents/SynSig_August2020/SynSig/prev_databases/SynGO_CC.txt')
	syngo_cc_genes=syngo.genes
	#syngo_genes=list(set(syngo_bp_genes+syngo_cc_genes))
	return syngo_cc_genes

def find_GO_synapse():
	df=pd.read_csv('/Users/karenmei/Documents/SynSig_August2020/SynSig/prev_databases/GO_Synapse.csv')
	genes=df['genes'].tolist()
	return genes

def clean_gene_names(genelist):
	cleaned=[]
	fine=[]
	for gene in genelist:
		if ";" in gene:
			print (gene)
			sublist=[x.strip() for x in gene.split(';')]
			print (sublist)
			cleaned.append(sublist)
		else:
			fine.append(gene)

	flat_list = [item for sublist in cleaned for item in sublist]
	print (flat_list)
	print (len(fine))

	full=list(set(fine+flat_list))
	print (len(full))
	return full

def find_synDB_genes():
	synDB=pd.read_csv('/Users/karenmei/Documents/SynSig_August2020/SynSig/prev_databases/SynDB_Master.csv')
	synDB_genes=synDB['Symbol'].tolist()
	df=pd.DataFrame({'genes': synDB_genes})
	df.to_csv('SynDB.csv')
	return synDB_genes

def find_synsysnet_genes():
	synsysnet=pd.read_csv('/Users/karenmei/Documents/SynSig_August2020/SynSig/prev_databases/SynSysNet_genes.csv')
	synsysnet_genes=synsysnet['gene_name'].tolist()
	df=pd.DataFrame({'genes': synsysnet_genes})
	df.to_csv('synsysnet.csv')
	return synsysnet_genes


syngo=load_syngo_genes()
print (len(syngo))
syngo=clean_gene_names(syngo)
corr_syngo=pd.DataFrame({'genes': syngo})
corr_syngo.to_csv('corr_syngo_cc.csv')

go=find_GO_synapse()
print (len(go))
go=clean_gene_names(go)
corr_go=pd.DataFrame({'genes': go})
corr_go.to_csv('corr_go.csv')

