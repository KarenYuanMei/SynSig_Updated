#Goal: to add function to predicted synapse gene (synsig) gene names
#files: kinases + human+reviewed from uniprot

import pandas as pd

import csv

import numpy


def load_synsig_genes():
	synsig=pd.read_csv('synsig_only.csv')
	#print (synsig)
	synsig_genes=synsig['genes'].tolist()
	return synsig_genes

def find_uniprot_genes(filename):
	df=pd.read_excel(filename)
	gene_names=df['Gene names'].tolist()
	all_gene_names=[]
	for item in gene_names:
		item=str(item)
		print (item)
		entries=item.split(' ')
		print (entries)
		all_gene_names.append(entries)
	flat_list = [item for sublist in all_gene_names for item in sublist]
	return flat_list

def find_function_cat(filename, name1, name2, gene_list, col_idx):
	synsig=pd.read_csv('synsig_only.csv')
	#print (synsig)
	synsig_genes=synsig['genes'].tolist()

	overlap=[]
	for gene in synsig_genes:
		if gene in gene_list:
			function=1
		else:
			function=0
		overlap.append(function)

	synsig.insert(loc=col_idx, column='%s/%s'%(name1, name2), value=overlap)
	print (synsig)

	synsig.to_csv('synsig_function.csv')
	return synsig

synsig_genes=load_synsig_genes()
filename='uniprot_kinases_human_reviewed.xlsx'
kinases=find_uniprot_genes(filename)
print (len(kinases))
print (kinases)

filename='uniprot_phosphatases_human_reviewed.xlsx'
phosphatases=find_uniprot_genes(filename)
print (len(phosphatases))
gene_list=kinases+phosphatases

synsig=find_function_cat('synsig_only.csv', 'kinases', 'phosphatases',gene_list, 2)

