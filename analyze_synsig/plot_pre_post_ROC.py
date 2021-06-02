#Goal: to show that predictions can recover both postsynaptic and presynaptic genes; Figure 2B

import pandas as pd
import numpy as np
import csv

import ddot
from ddot import Ontology

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 14})

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles



import sys
sys.path.append('../read_data_functions/')
import load_data_functions

import sys
sys.path.append('../graph_functions/')
import graph_functions

sys.path.append('../ML_functions/')
import find_training_genes_functions
import find_GO_scores
import ROC_functions


def load_propagated_syngo():
	syngo=Ontology.from_table('../source_data_files/correct_db/SynGO_CC.txt')

	syngo = syngo.propagate(direction='forward', gene_term=True, term_term=False)
	return syngo

def find_focus_terms_genes(ont, branch_name):

	branch=ont.focus(branches=branch_name)

	branch_genes=branch.genes
	branch_terms=branch.terms
	return branch_genes, branch_terms

def draw_venn_2(genelist1, genelist2, label_list):
	f = plt.figure()
	venn2([set(genelist1), set(genelist2)], set_labels = (label_list[0], label_list[1]), set_colors=('red', 'skyblue'), alpha = 0.5)
	plt.show()
	f.savefig("pre_post_terms_overlap.svg", bbox_inches='tight')

big_pool=load_data_functions.load_big_pool()

all_training=find_training_genes_functions.load_pos_neg_training()

human_ont=find_GO_scores.find_GO_ont()
go_genes=human_ont.genes

syngo_ont=load_propagated_syngo()
post_genes, post_terms=find_focus_terms_genes(syngo_ont, 'postsynapse')
pre_genes, pre_terms=find_focus_terms_genes(syngo_ont, 'presynapse')


label_list=['Presyn Genes', 'Postsyn Genes']


draw_venn_2(post_genes, pre_genes, label_list)
draw_venn_2(post_terms, pre_terms, label_list)


# training_post=list(set(training)&set(post_genes))
# print (len(training_post))

# training_pre=list(set(training)&set(pre_genes))
# print (len(training_pre))

# test_pos=list(set(list(set(post_genes)-set(training_post)))&set(GO_genes))
# print (len(test_pos))

# test_pre=list(set(list(set(pre_genes)-set(training_pre)))&set(GO_genes))
# print (len(test_pre))

# overlap=list(set(test_pos)&set(test_pre))
# print (len(overlap))

db_list=[pre_genes, post_genes]
db_labels=['Presynapse', 'Postsynapse']

pred_df=load_data_functions.load_predicted_synsig_df()

for i in range(len(db_list)):
	final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, db_list[i], all_training)
	fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)
	print (auc)
	ROC_functions.save_roc_df(thresholds, tpr, fpr, i, db_labels[i])

	#make Figure 2A
	graph_functions.plot_tandem_ROC(tpr, fpr, auc,'pre_vs_post')



