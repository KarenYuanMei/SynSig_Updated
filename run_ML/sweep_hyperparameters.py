#Goal: optimize random forest hyperparameters


import pandas as pd
import numpy as np
import csv
import random

import ddot
from ddot import Ontology

from itertools import combinations, combinations_with_replacement
from itertools import product
from collections import defaultdict

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

sys.path.append('../ML_functions/')
import define_gene_objects
import regressor_functions
import find_training_genes_functions
import find_GO_scores

import time

#few_plus_one_features
def define_final_features():
	feature_list=['cerebral_cortex_hpa_isoform_exp', 'gtex_rna_tissue_expression','Phosphosite_hu_no', 'pFAM_domain_number', 'pFAM_domain', 'protein_mass', 'Ensembl_aa_length', 'Ensembl_isoform_no', 'trans_count', 'gc_content', 'trans_len', 'gene_length', 'exon_no', 'cds_length']

	brain_features=['HIP_RNA', 'DFC_RNA', 'V1C_RNA', 'AMY_RNA', 'MD_RNA', 'STR_RNA', 'CBC_RNA']
	kernel_feature=['mentha_kernel']
	#gtex_kernel_feature=['gtex_rna_kernel']
	feature_list=feature_list+brain_features+kernel_feature
	#feature_list.pop(idx)

	return feature_list


big_pool=find_training_genes_functions.load_big_pool()

pos, neg, all_training=find_training_genes_functions.load_pos_neg_training()

pos_chunks, neg_chunks=find_training_genes_functions.find_pos_neg_chunks(pos, neg)

for i in range(5):
	training_gene_names, test_gene_names=find_training_genes_functions.define_training_test(pos, pos_chunks, neg, neg_chunks, i)
	training_df=find_training_genes_functions.make_genes_csv(training_gene_names, 'new', 'training_genes_%s'%i)
	test_df=find_training_genes_functions.make_genes_csv(test_gene_names, 'new', 'test_genes_%s'%i)

feature_value_dict = define_gene_objects.create_feature_value_dict(big_pool)
print ("DONE1")

all_training_objects=define_gene_objects.define_all_training_objects(all_training, go_mat_filename, feature_value_dict)

feature_list=define_final_features()

