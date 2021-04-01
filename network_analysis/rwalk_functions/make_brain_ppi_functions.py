#Goal: to stitch together the Coba 2017 brain ppis into a single network


import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt

import make_network_graph_functions

#print (df)


def make_brain_df(sheetno):
	df=pd.read_excel('../../source_data_files/ppi_files/coba_2017_brain.xlsx', sheetname=sheetno)
	df=df.apply(lambda x: x.astype(str).str.upper())
	node1=df['Protein'].tolist()
	node2=df['Interactor'].tolist()
	new_df=pd.DataFrame({'Node1':node1, 'Node2': node2})
	return new_df

def make_brain_network():
	sheets=[0, 1]
	dfs=[]
	for sheet in sheets:
		new_df=make_brain_df(sheet)
		print (new_df)
		dfs.append(new_df)

	brain_ppi=pd.concat(dfs)
	print (brain_ppi)

	G=make_network_graph_functions.make_network_G(brain_ppi)
	return G
