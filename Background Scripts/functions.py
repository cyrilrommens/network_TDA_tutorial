# List of all the necessary functions

# Import libraries
import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import glob
import timeit
import os

# Define function to draw a visual graph from the connection matrix
def plot_graph(correlation_matrix, threshold):
    n = correlation_matrix.shape[0]
    G = nx.Graph()
    
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)  # You can use a different layout if needed
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700)
    plt.show()

# Define a function to obtain a list the simplexes present in the simplicial complex, by counting the complete subgraphs in the connection matrix
def build_clique_complex(correlation_matrix, threshold, max_clique_size):
    n = correlation_matrix.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)

    # Using nx.enumerate_all_cliques in an interactive manner
    seen_cliques = set()
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) > max_clique_size:
            break
        unique_clique = tuple(sorted(clique))
        seen_cliques.add(unique_clique)

    # Building the clique complex
    clique_complex = [set(clique) for clique in seen_cliques]

    # Sort the list of sets based on the length of cliques and sorted vertices within each clique
    clique_complex = sorted(clique_complex, key=lambda x: (len(x), sorted(x)))

    return clique_complex

# Define a function to generate a connection matrix of the simplexes from the list of cliques
def generate_overlap_matrix(sets_list):
    # Get the set of all unique values in the list of sets
    all_values = sorted(list({value for s in sets_list for value in s}))
    
    # Create a mapping from values to indices
    value_to_index = {value: index for index, value in enumerate(all_values)}
    
    # Initialize the overlap matrix with zeros
    n = len(sets_list)
    overlap_matrix = np.zeros((n, n), dtype=int)
    
    # Set the entries to 1 where values overlap
    for i, s1 in enumerate(sets_list):
        values_s1 = sorted(list(s1))
        indices_s1 = [value_to_index[value] for value in values_s1]
        
        for j, s2 in enumerate(sets_list):
            values_s2 = sorted(list(s2))
            indices_s2 = [value_to_index[value] for value in values_s2]
            
            if any(value in s2 for value in s1):  # Check for overlap
                overlap_matrix[i, j] = 1
    
    return overlap_matrix