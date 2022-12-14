#Goal: 1) analyze the predictions on the new genes from using ALL of SynGO

import pandas as pd
import csv
import random

import ddot
from ddot import Ontology

from itertools import combinations, combinations_with_replacement
from itertools import product
from collections import defaultdict

import matplotlib.pyplot as plt

import sys
sys.path.append('../read_data_functions/')
import load_data_functions

#sys.path.append('../ML_functions/')
#import find_training_genes_functions
#import find_GO_scores
#import ROC_functions


#find the AUROC with the consensus mass spectrometry findings: ========================================================
#load the predicted df:
pred_df=pd.read_csv('../run_ML/ML_output/all_syngo_brain_RNA_big_pool_novel_synapse_genes_avg_scores.csv', index_col=[0])
print (pred_df)

# #load the human ont genes:
# human_ont=find_GO_scores.find_GO_ont()
# go_genes=human_ont.genes


# big_pool=load_data_functions.load_big_pool()

# syngo=load_data_functions.find_syngo(big_pool, go_genes)

# syndb=load_data_functions.find_SynDB(big_pool)
# synsysnet=load_data_functions.find_synsysnet(big_pool)

# ctx_genes=load_data_functions.find_adult_cortex(big_pool)
# str_genes=load_data_functions.find_adult_striatum(big_pool)
# fetal_genes=load_data_functions.find_fetal(big_pool)
# ngn2_genes=load_data_functions.find_ngn2(big_pool)

# consensus_ms=list(set(ctx_genes)&set(str_genes)&set(fetal_genes)&set(ngn2_genes))
# print (len(consensus_ms))
# #528 consensus mass spectrometry genes

# consensus_no_syngo=list(set(consensus_ms)-set(syngo))
# print (len(consensus_no_syngo))
# #328 consensus mass spectrometry genes; after filtering out syngo genes

# final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, consensus_ms, syngo)
# fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)

# print (auc)
# #0.847

# #save the ROC tpr, fpr, thresholds to csv
# ROC_functions.save_roc_df(thresholds, tpr, fpr, 1, 'all_syngo')
# #output for the mass spectrometry consensus evaluation:
# #all_syngo_cv_roc_df_1.csv


# consensus_syn=list(set(syndb)&set(synsysnet))
# consensus_syn=list(set(consensus_syn)-set(syngo))
# print (len(consensus_syn))
# #276 genes

# final, label, avg_score=ROC_functions.find_pred_labels_scores(pred_df, consensus_syn, syngo)
# fpr, tpr, thresholds, auc=ROC_functions.calculate_roc(label, avg_score)

# print (auc)
# #0.843

#save the ROC tpr, fpr, thresholds to csv
#ROC_functions.save_roc_df(thresholds, tpr, fpr, 1, 'all_syngo_syn')


#Compare the previous prediction ranking with the new prediction ranking: =================================
prev_df_file='~/Documents/SynSig_August2020/revisions/SynSig_Updated/run_ML/update_web_table.csv'
prev_df=pd.read_csv(prev_df_file, index_col=[0])

print ('PREVIOUS', prev_df)

prev_genes=prev_df['genes'].tolist()
new_genes=pred_df['genes'].tolist()

prev_df=prev_df.set_index(['genes'])
print ('new previous', prev_df)
pred_df=pred_df.set_index(['genes'])

overlap=list(set(prev_genes)&set(new_genes))
print ('overlap', len(overlap))

overlap_prev=prev_df.loc[overlap]
print ('Overlap previous', overlap_prev)
prev_rank=overlap_prev['avg_scores'].tolist()

overlap_new=pred_df.loc[overlap]
print ('overlap new', overlap_new)
new_rank=overlap_new['avg_scores'].tolist()

from scipy import stats

print (stats.spearmanr(prev_rank, new_rank))

#SpearmanrResult(correlation=0.9255386636994402, pvalue=0.0)

plt.scatter(prev_rank, new_rank, c='darkred')
plt.xlabel('60% SynGO Predicted Synapse Similarity Scores')
plt.ylabel('100% SynGO Predicted Synapse Similarity Scores')

#plt.xlim(0, 7)
#plt.ylim(0,7)
plt.savefig('60syngo_vs_allsyngo_pred.svg')





