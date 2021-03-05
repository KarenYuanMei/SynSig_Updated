#Goal: use the environment net-pop to run random walk

import scipy as sp
import networkx as nx
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.append('../rwalk_functions/')
import make_network_graph_functions

sys.path.append('../ppi_files/')

filename='../ppi_files/Human_Mentha_converted.csv'
df=make_network_graph_functions.make_mentha_df(filename)
print (df)

#print (nx.test())

G=make_network_graph_functions.make_network_G(df)
print (G.number_of_edges())