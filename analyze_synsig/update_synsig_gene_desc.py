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

	#print (d['GO:0000035'])
	return d

desc_df=pd.read_csv('../source_data_files/gene_names/gene_name_description.txt')
#print (desc_df)

#make gene_name: gene_description dictionary
names=desc_df['Gene name'].tolist()
desc=desc_df['Gene description'].tolist()
names_desc=list(zip(names, desc))
names_desc_dict=dict(names_desc)
#print (names_desc_dict['MT-TF'])

synsig_df=pd.read_csv('../run_ML/update_web_table.csv', index_col=[0])
#print (synsig_df)

synsig_genes=synsig_df['genes'].tolist()

synsig_desc_list=[]
for item in synsig_genes:
	if item in names:
		synsig_desc=names_desc_dict[item]
	else:
		synsig_desc='not found'
	synsig_desc_list.append(synsig_desc)

#print (len(synsig_genes))
#print (len(synsig_desc_list))

idx=1

synsig_df.insert(loc=idx, column='description', value=synsig_desc_list)

#synsig_df['description']=synsig_desc_list
#print (synsig_df)

synsig_df.to_csv('update_web_table_desc.csv')

synsig=synsig_df[synsig_df['SynSig']=='yes']
#print (synsig)

synsig.to_csv('synsig_only.csv')

synsig=pd.read_csv('synsig_only.csv', index_col=[0])
gene_desc=synsig['description'].tolist()


def find_function_cat(filename, names, col_idx, cat_name):
	synsig=pd.read_csv(filename, index_col=[0])
	gene_desc=synsig['description'].tolist()

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
	print (num)

	synsig.insert(loc=col_idx, column=cat_name, value=functions)
	#print (synsig)

	synsig.to_csv('synsig_function.csv')
	return synsig


synsig=find_function_cat('synsig_only.csv', ['receptor', 'channel'], 2, 'receptor/channel')
synsig=find_function_cat('synsig_function.csv', ['kinase'], 3, 'kinase')
synsig=find_function_cat('synsig_function.csv', ['phosphatase'], 4, 'phosphatase')
synsig=find_function_cat('synsig_function.csv', ['ubiquitin', 'E3'], 5, 'ubiquitin/E3')
synsig=find_function_cat('synsig_function.csv', ['membrane', 'transmembrane', 'symporter'], 6, 'membrane')
synsig=find_function_cat('synsig_function.csv', ['GTPase', 'ATPase', 'exchange factor', 'GTP', 'ATP'], 7, 'GTP/ATP regulators')
synsig=find_function_cat('synsig_function.csv', ['DNA ', 'chromatin', 'transcription', 'nucleic acid', 'nucleotide'], 8, 'Nucleic Acid Binding')
synsig=find_function_cat('synsig_function.csv', ['ribosom', 'translation', 'RNA', 'helicase'], 9, 'translation')
synsig=find_function_cat('synsig_function.csv', ['translocase', 'export', 'import', 'transport', 'myosin', 'kinesin', 'dynein', 'dynactin'], 10, 'transport')
synsig=find_function_cat('synsig_function.csv', ['adhesion', 'cadherin', 'junction', 'catenin'], 11, 'cell adhesion')
synsig=find_function_cat('synsig_function.csv', ['heat shock', 'regulator', 'adaptor', 'chaperone'], 12, 'regulators')
synsig=find_function_cat('synsig_function.csv', ['scaffold', 'assembl', 'adaptor'], 12, 'scaffolds/adaptors')
synsig=find_function_cat('synsig_function.csv', ['microtubule', 'actin', 'filament', 'tubulin', 'filamin', 'cytoskelet'], 13, 'cytoskeletal')
synsig=find_function_cat('synsig_function.csv', ['calcium ion', 'calmodulin binding'], 14, 'calcium ion binding')



synsig['Function Total']= synsig.iloc[:, 2:15].sum(axis=1)

func_total=synsig['Function Total'].tolist()
synsig.pop('Function Total')
synsig.insert(loc=15, column='Function Total', value=func_total)
#print (synsig)
synsig.to_csv('synsig_function.csv')

no_func=synsig[synsig['Function Total']==0]
print (no_func)
no_func.to_csv('no_func.csv')
no_func_genes=no_func['genes'].tolist()
print (len(no_func_genes))

go_human=find_GO_scores.find_GO_ont()
#print (go_human)

terms=go_human.terms
#print (terms)

child_terms=go_human.parent_2_child['GO:00SUPER']
print (child_terms)
mf=child_terms[0]
print (mf)
ont2 = go_human.focus(branches=mf)
print (ont2)

terms=ont2.terms

overlap=list(set(no_func_genes)&set(ont2.genes))
print (len(overlap))

d=make_goID_dict()

all_term_names=[]
for gene in overlap:
	gene_term_names=[]
	gene_terms=ont2.gene_2_term[gene]
	for item in gene_terms:
		term_desc=terms[item]
		term_name=d[term_desc]
		gene_term_names.append(term_name)
		print (gene, gene_term_names)
	all_term_names.append(gene_term_names)

df=pd.DataFrame({'genes': overlap, 'terms': all_term_names})
df.to_csv('no_func_mf.csv')
print (df)


# def find_function_cat(filename, names, col_idx, cat_name):
# 	synsig=pd.read_csv(filename, index_col=[0])
# 	gene_desc=synsig['terms'].tolist()

# 	functions=[]
	
# 	for item in gene_desc:
# 		names_in_desc = any(name in item for name in names)
# 		#print (names_in_desc)
# 		if names_in_desc==False:
# 			function=0
# 		else:
# 			function=1

# 		functions.append(function)

# 	num=functions.count(1)
# 	print (cat_name, num)

# 	synsig.insert(loc=col_idx, column=cat_name, value=functions)
# 	#print (synsig)

# 	synsig.to_csv('synsig_no_func_mf.csv')
# 	return synsig

# synsig=find_function_cat('no_func_mf.csv', ['receptor', 'channel'], 2, 'receptor/channel')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['kinase'], 3, 'kinase')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['phosphatase'], 4, 'phosphatase')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['ubiquitin', 'E3'], 5, 'ubiquitin/E3')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['membrane', 'transmembrane', 'symporter'], 6, 'membrane')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['GTPase', 'ATPase', 'exchange factor', 'GTP', 'ATP'], 7, 'GTP/ATP regulators')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['DNA ', 'chromatin', 'transcription', 'nucleic acid', 'nucleotide'], 8, 'Nucleic Acid Binding')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['ribosom', 'translation', 'RNA', 'helicase'], 9, 'translation')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['translocase', 'export', 'import', 'transport', 'myosin', 'kinesin', 'dynein', 'dynactin'], 10, 'transport')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['adhesion', 'cadherin', 'junction', 'catenin'], 11, 'cell adhesion')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['heat shock', 'regulator', 'adaptor', 'chaperone'], 12, 'regulators')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['scaffold', 'assembl', 'adaptor'], 12, 'scaffolds/adaptors')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['microtubule', 'actin', 'filament', 'tubulin', 'filamin', 'cytoskelet'], 13, 'cytoskeletal')
# synsig=find_function_cat('synsig_no_func_mf.csv', ['calcium ion', 'calmodulin binding'], 14, 'calcium ion binding')


# print (synsig)

# synsig['Function Total']= synsig.iloc[:, 2:15].sum(axis=1)
# no_func=synsig[synsig['Function Total']==0]
# print (no_func)
# synsig.to_csv('synsig_no_func_mf_total.csv')

# no_func.to_csv('synsig_no_func_manual.csv')

