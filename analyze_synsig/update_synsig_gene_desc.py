#Goal: to add description to predicted synapse gene (synsig) gene names
#gene_name_description.txt was downloaded on April 9th from Biomart

import pandas as pd

import csv

import numpy

import ddot
from ddot import Ontology

import sys
sys.path.append('../ML_functions/')
import find_GO_scores

from collections import defaultdict

def make_goID_dict():
	id_name=pd.read_table('../source_data_files/goID/goID_2_name.tab')
	print (id_name)

	id_name.columns=['ID', 'name']
	print (id_name)

	ids=id_name['ID'].tolist()
	names=id_name['name'].tolist()

	id_name_zip=list(zip(ids, names))
	d=defaultdict(list)

	for term, name in id_name_zip:
		d[term].append(name)
	return d

def find_function_cat(filename, variable, names, col_idx, cat_name, csv_filename):
	synsig=pd.read_csv(filename, index_col=[0])
	gene_desc=synsig[variable].tolist()

	functions=[]
	
	for item in gene_desc:
		names_in_desc = any(name in item for name in names)
		#print (names_in_desc)
		if names_in_desc==False:
			function=0
		else:
			function=1

		functions.append(function)

	num=functions.count(1)
	print (cat_name, num)

	synsig.insert(loc=col_idx, column=cat_name, value=functions)
	synsig.to_csv('%s'%csv_filename)
	return synsig


def make_gene_desc_dict():

	desc_df=pd.read_csv('../source_data_files/gene_names/gene_name_description.txt')

	#make gene_name: gene_description dictionary
	names=desc_df['Gene name'].tolist()
	desc=desc_df['Gene description'].tolist()
	names_desc=list(zip(names, desc))
	names_desc_dict=dict(names_desc)
	return names_desc_dict

def add_gene_desc_web(names_desc_dict, genelist):

	names=list(names_desc_dict.keys())

	synsig_desc_list=[]
	for item in genelist:
		if item in names:
			synsig_desc=names_desc_dict[item]
		else:
			synsig_desc='not found'
		synsig_desc_list.append(synsig_desc)

	idx=1

	#synsig_df.insert(loc=idx, column='description', value=synsig_desc_list)

	#synsig_df.to_csv('update_web_table_desc.csv')
	synsig_df=pd.DataFrame({'genes': synsig_genes, 'description': synsig_desc_list})
	
	return synsig_df

def find_synsig_only(synsig_df):
	synsig=synsig_df[synsig_df['SynSig']=='yes']
	synsig.to_csv('synsig_only.csv')
	return synsig

def annotate_function(filename1, variable, filename2):
	synsig=find_function_cat(filename1, variable, ['receptor', 'channel'], 2, 'receptor/channel', filename2)
	synsig=find_function_cat(filename2, variable, ['kinase'], 3, 'kinase', filename2)
	synsig=find_function_cat(filename2, variable, ['phosphatase'], 4, 'phosphatase', filename2)
	synsig=find_function_cat(filename2, variable, ['ubiquitin', 'E3'], 5, 'ubiquitin/E3', filename2)
	synsig=find_function_cat(filename2, variable, ['membrane', 'transmembrane', 'symporter'], 6, 'membrane', filename2)
	synsig=find_function_cat(filename2, variable, ['GTPase', 'ATPase', 'exchange factor', 'GTP', 'ATP'], 7, 'GTP/ATP regulators', filename2)
	synsig=find_function_cat(filename2, variable, ['DNA ', 'chromatin', 'transcription', 'nucleic acid', 'nucleotide'], 8, 'Nucleic Acid Binding', filename2)
	synsig=find_function_cat(filename2, variable, ['ribosom', 'translation', 'RNA', 'helicase'], 9, 'translation', filename2)
	synsig=find_function_cat(filename2, variable, ['translocase', 'export', 'import', 'transport', 'myosin', 'kinesin', 'dynein', 'dynactin'], 10, 'transport', filename2)
	synsig=find_function_cat(filename2, variable, ['adhesion', 'cadherin', 'junction', 'catenin'], 11, 'cell adhesion', filename2)
	synsig=find_function_cat(filename2, variable, ['heat shock', 'regulator', 'chaperone'], 12, 'regulators', filename2)
	synsig=find_function_cat(filename2, variable, ['scaffold', 'assembl', 'adaptor'], 13, 'scaffolds/adaptors', filename2)
	synsig=find_function_cat(filename2, variable, ['microtubule', 'actin', 'filament', 'tubulin', 'filamin', 'cytoskelet'], 14, 'cytoskeletal', filename2)
	synsig=find_function_cat(filename2, variable, ['calcium ion', 'calmodulin binding'], 15, 'calcium ion binding', filename2)

	synsig['Function Total']= synsig.iloc[:, 2:16].sum(axis=1)

	func_total=synsig['Function Total'].tolist()
	synsig.pop('Function Total')
	synsig.insert(loc=16, column='Function Total', value=func_total)
	#print (synsig)
	synsig.to_csv(filename2)
	return synsig

#Annotate function by gene description: first wrapper function:
def annotate_function_by_desc():
	names_desc_dict=make_gene_desc_dict()
	synsig_df=add_gene_desc_web(names_desc_dict)
	synsig=find_synsig_only(synsig_df)

	filename1='synsig_only.csv'
	filename2='synsig_function.csv'

	synsig=annotate_function(filename1,'description', filename2)
	return synsig


def find_unannotated_genes(df):
	no_func=df[df['Function Total']==0]
	print (no_func)
	no_func.to_csv('no_func.csv')
	no_func_genes=no_func['genes'].tolist()
	print (len(no_func_genes))
	return no_func_genes

def find_mf_ont():
	go_human=find_GO_scores.find_GO_ont()
	child_terms=go_human.parent_2_child['GO:00SUPER']
	mf=child_terms[0]
	ont2 = go_human.focus(branches=mf)
	return ont2

def add_mf_function(genelist, synsig_desc):
	ont2=find_mf_ont()
	terms=ont2.terms

	overlap=list(set(genelist)&set(ont2.genes))
	print (len(overlap))

	d=make_goID_dict()

	all_term_names=[]
	for gene in genelist:
		if gene in overlap:
			gene_term_names=[]
			gene_terms=ont2.gene_2_term[gene]
			for item in gene_terms:
				term_desc=terms[item]
				term_name=d[term_desc]
				gene_term_names.append(term_name)
				
		else:
			gene_term_names='None'
		all_term_names.append(gene_term_names)

	synsig_desc.insert(loc=2, column='MF Terms', value=all_term_names)
	return synsig_desc

def add_desc(synsig_genes):
	names_desc_dict=make_gene_desc_dict()
	synsig_desc=add_gene_desc_web(names_desc_dict, synsig_genes)
	print (synsig_desc)
	return synsig_desc

#add description to all synsig genes:
def load_synsig_genes():
	synsig_df=pd.read_csv('../run_ML/update_web_table.csv', index_col=[0])
	#print (synsig_df)
	synsig=synsig_df[synsig_df['SynSig']=='yes']
	synsig_genes=synsig['genes'].tolist()
	return synsig_genes

synsig_genes=load_synsig_genes()

#add description all synsig genes:
synsig_desc=add_desc(synsig_genes)
synsig_desc.to_csv('synsig_genes_desc.csv')

#add mf to all synsig genes:
synsig_desc_mf=add_mf_function(synsig_genes, synsig_desc)
synsig_desc_mf.to_csv('synsig_desc_mf.csv')
print (synsig_desc_mf)

#first annotate genes by gene description:







# synsig_by_desc=annotate_function_by_desc()

# no_func_genes=find_unannotated_genes(synsig_by_desc)

# synsig_no_func_mf=add_mf_function(no_func_genes)

# added_func_df=annotate_function('no_func_mf.csv', 'terms', 'synsig_additional_func_mf.csv')

# no_func=find_unannotated_genes(added_func_df)

