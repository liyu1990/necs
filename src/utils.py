#!/usr/bin/env python
# coding:utf-8
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd


def cosine_similarity(G):
    degrees = nx.degree(G)
    sets = {node: set(G.neighbors(node)) for node in nx.nodes(G)}
    laps = np.array(
            [[float(len(sets[node_1].intersection(sets[node_2]))) / (float(degrees[node_1] * degrees[node_2]) ** 0.5) 
                if node_1 != node_2 else 0.0 for node_1 in nx.nodes(G)] 
                for node_2 in tqdm(nx.nodes(G))], 
            dtype=np.float64)
    return laps


def graph_reader(input_path):
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

