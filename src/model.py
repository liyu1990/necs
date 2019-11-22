#!/usr/bin/env python
# coding:utf-8
# @author : liyu
import os
import math
import numpy as np
import networkx as nx
from utils import cosine_similarity
from utils import graph_reader
from copy import deepcopy


class NECS(object):
    def __init__(self, name, dimensions=128, alpha=0.1, beta=1, times=1, clusters=5, decay=0.1, order=3, max_iteration=300):
        print("Model initialization started.\n")
        self.input = u"../data/{}.csv".format(name)
        self.name = name
        self.lambd = math.pow(10, 8)
        self.dimensions = dimensions
        self.alpha = alpha
        self.beta = beta
        self.times = times
        self.clusters = clusters
        self.order = order
        self.decay = decay
        self.max_iteration = max_iteration
        self.converge_threshold = math.pow(10, -3)
        self.lower_control = math.pow(10, -8)

        self.G = graph_reader(self.input)
        self.number_of_nodes = len(nx.nodes(self.G))

        self.S = cosine_similarity(self.G)  # cosine similarity
        self.Adj = np.array(nx.adjacency_matrix(self.G).todense())  # adjacency Matrix
        self.P = self.high_order_proximity()  # high-order proximity

        self.current_loss = math.pow(10, 10)
        self.round = 0
        self.V = self.matrix_random_initialization(self.number_of_nodes, self.dimensions)
        self.U = self.matrix_random_initialization(self.number_of_nodes, self.dimensions)
        self.H = self.matrix_random_initialization(self.number_of_nodes, self.clusters)
        self.W = self.matrix_random_initialization(self.clusters, self.dimensions)

    def high_order_proximity(self):
        A_t = np.ones_like(self.Adj, dtype=np.float64)
        P = np.zeros_like(self.Adj, dtype=np.float64)
        for i in range(self.order):
            if i == 0:
                A_t = deepcopy(self.Adj)
            else:
                A_t = np.matmul(deepcopy(A_t), self.Adj)
            P += np.float64(self.decay ** i) * A_t
        return P

    def re_matrix_random_initialization(self):
        self.current_loss = math.pow(10, 10)
        self.round = 0
        self.P = self.high_order_proximity()
        self.V = self.matrix_random_initialization(self.number_of_nodes, self.dimensions)
        self.U = self.matrix_random_initialization(self.number_of_nodes, self.dimensions)
        self.H = self.matrix_random_initialization(self.number_of_nodes, self.clusters)
        self.W = self.matrix_random_initialization(self.clusters, self.dimensions)

    @staticmethod
    def matrix_random_initialization(n_components, n_features):
        return np.random.uniform(0, 1, [n_components, n_features])

    def update_rules(self):
        # Update V

        numerator_v = np.matmul(self.P, self.U)
        denominator_v = np.matmul(self.V, np.matmul(self.U.T, self.U))
        denominator_v = np.maximum(np.float64(self.lower_control), denominator_v)
        self.V = np.multiply(self.V, np.sqrt(numerator_v / denominator_v))

        # Update U

        numerator_u = np.matmul(self.P.T, self.V) + self.beta * np.matmul(self.H, self.W)
        denominator_u = np.matmul(self.U, np.matmul(self.V.T, self.V) + self.beta * np.matmul(self.W.T, self.W))
        denominator_u = np.maximum(np.float64(self.lower_control), denominator_u)
        self.U = np.multiply(self.U, np.sqrt(numerator_u / denominator_u))

        # Update W

        numerator_w = np.matmul(self.H.T, self.U)
        denominator_w = np.matmul(self.W, np.matmul(self.U.T, self.U))
        denominator_w = np.maximum(np.float64(self.lower_control), denominator_w)
        self.W = np.multiply(self.W, np.sqrt(numerator_w / denominator_w))

        # Update H

        self.SH = np.matmul(self.S, self.H)
        self.HHH = np.matmul(self.H, np.matmul(self.H.T, self.H))
        self.UW = np.matmul(self.U, self.W.T)
        numerator_h = np.multiply(
            (self.lambd + self.alpha) * self.HHH,
            4 * self.alpha * self.SH + 2 * self.beta * self.UW + (4 * self.lambd - 2 * self.beta) * self.H)
        numerator_h = np.sqrt(numerator_h)
        denominator_h = 2 * (self.lambd + self.alpha) * self.HHH
        denominator_h = np.maximum(np.float64(self.lower_control), denominator_h)
        self.H = np.multiply(self.H, np.sqrt(numerator_h / denominator_h))

    def loss(self):
        item1 = np.linalg.norm(self.P - np.matmul(self.V, self.U.T), ord="fro") ** 2
        item2 = self.alpha * (np.linalg.norm(self.S - np.matmul(self.H, self.H.T), ord="fro") ** 2)
        item3 = self.beta * (np.linalg.norm(self.H - np.matmul(self.U, self.W.T), ord="fro") ** 2)

        constraint = self.lambd * (
                np.linalg.norm(np.matmul(self.H.T, self.H) - np.eye(self.clusters), ord="fro") ** 2)
        loss = np.float64(item1 + item2 + item3 + constraint)
        print("{}\t{}\t{}\t{}\t{}\t{}".format(self.round, item1, item2, item3, constraint, loss))
        return loss

    def save_matrix(self, ):
        dst_fold = "../output/{}/{}/{}".format(
            self.name,
            "Dim{}Alpha{}Beta{}Order{}Decay{}".format(self.dimensions, self.alpha, self.beta, self.order, self.decay),
            self.times)
        if not os.path.exists(dst_fold):
            os.makedirs(dst_fold)
        np.savetxt(os.path.join(dst_fold, 'U.txt'), self.U, fmt='%f', delimiter=',')

    def run(self):
        for epoch in range(self.max_iteration):
            self.round = epoch + 1
            self.update_rules()
            loss = self.loss()
            if abs(self.current_loss - loss) < self.converge_threshold:
                break
            if loss < self.current_loss:
                self.current_loss = loss

        self.save_matrix()
