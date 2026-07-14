#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import random
from numpy.linalg import matrix_power
import gurobipy as grb
import time


def positive_adjacency_from_signed(S):
    #create a positive adjacency matrix from signed matrix S
    pos_adj_mx = (S == 1).astype(int)
    #every vertex is included in its own positive neighborhood
    np.fill_diagonal(pos_adj_mx, 1)
    return pos_adj_mx

def add_min_max_vertex_constraints(M, x, S):
    n = np.shape(S)[0]
    for v in range(n):
        cons = grb.LinExpr()
        for w in range(n):
            if w == v:
                continue
            #deleted edge contributes nothing
            if S[v,w] == 0:
                continue
            if w < v:
                edge_var = x[w,v]
            else:
                edge_var = x[v,w]
            #positive edge contributes x_vw
            if S[v,w] == 1:
                cons = cons + edge_var
            #negative edge contributes 1 - x_vw
            elif S[v,w] == -1:
                cons = cons + 1 - edge_var
        M.addConstr(x[0,0] >= cons)

#Computing the correlation metric distances
#input: signed matrix, radii r and r2 for rounding algorithm
#output: correlation metric distances, L_0 values, r and r2 neighborhoods, time, fractional cost
def exact(S, r, r2):
    t0 = time.time()
    #create positive adjacency matrix for positive-neighborhood computations
    pos_adj_mx = positive_adjacency_from_signed(S)
    n = np.shape(S)[0]
    if not(np.array_equal(np.diagonal(pos_adj_mx), np.ones([n]))):
        raise Exception('Diagonal not all 1s')
    #initialize a dictionary that stores the L_t values of each vertex
    L_t_vals = {}
    #initialize the dictionaries that store the r and r2 neighborhoods of each vertex
    neighborsR = {}
    neighborsR2 = {}
    for k in range(n):
        L_t_vals.update({k: 0})
        neighborsR.update({k: []})
        neighborsR2.update({k: []})
    #for each pair of nodes, compute common positive neighborhood
    pos_len_2 = matrix_power(pos_adj_mx, 2)
    #compute the vector of positive degrees
    pos_degrees = np.matmul(pos_adj_mx, np.ones(n))
    #initialize the correlation metric distances
    distances = np.zeros([n,n])
    #compute exact correlation metric using sizes of positive neighborhoods
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            distances[u][v] = 1 - (
                (pos_len_2[u][v]) /
                (
                    pos_degrees[u]
                    + pos_degrees[v]
                    - pos_len_2[u][v]
                )
            )
            distances[v][u] = distances[u][v]
            #add to r2-neighborhood
            if distances[u][v] <= r2:
                neighborsR2[u].append(v)
                neighborsR2[v].append(u)
                #update L_t values and add to r-neighborhood
                if distances[u][v] <= r:
                    L_t_vals[u] = L_t_vals[u] + r - distances[u][v]
                    neighborsR[u].append(v)
                    L_t_vals[v] = L_t_vals[v] + r - distances[u][v]
                    neighborsR[v].append(u)
    t1 = time.time()
    clock = t1 - t0
    #for analysis only: compute the fractional cost of the correlation metric
    frac_values = []
    for v in range(n):
        tot = 0
        for w in range(n):
            if w == v:
                continue
            #negative edge contributes 1 - distance
            if S[v,w] == -1:
                tot = tot + 1 - distances[v][w]
            #positive edge contributes distance
            if S[v,w] == 1:
                tot = tot + distances[v][w]
            #deleted edge contributes nothing
        frac_values.append(tot)
    frac_val = max(frac_values)

    return distances, L_t_vals, neighborsR, neighborsR2, clock, frac_val

#KMZ Phase 2 (Rounding Algorithm)
#input: distances, L_0 values, R and R2 neighborhoods, radii r and r2
#output: set of clusters (as a list of lists) and time
def cluster(distances, L_t_vals, neighborsR, neighborsR2, r, r2):
    t0 = time.time()
    #store the clusters in this list
    clustering = []
    n = np.shape(distances)[0]
    #yet unclustered vertices
    num_unclustered = n
    #indicator list indexed by the vertices: 1 if unclustered, 0 if clustered
    V_t = np.ones([n])
    while num_unclustered > 0:
        #find the vertex maximizing L_t
        max_key = max(L_t_vals, key=L_t_vals.get)
        #initialize new cluster with maximizing vertex
        cluster = [max_key]
        #cut out the new cluster
        for j in range(len(neighborsR2[max_key])):
            #check whether this R2-neighbor of the maximizing vertex has been clustered yet
            if V_t[neighborsR2[max_key][j]] == 1:
                #if not, add to present cluster
                cluster.append(neighborsR2[max_key][j])
        #append the new cluster to the list of clusters
        clustering.append(cluster)
        #update the number of unclustered vertices remaining
        num_unclustered = num_unclustered - len(cluster)
        #update L_t
        for k in range(len(cluster)):
            #remove L_t values for clustered vertices
            del L_t_vals[cluster[k]]
            #mark clustered vertices as clustered
            V_t[cluster[k]] = 0
            #update remaining L_t values
            for key in L_t_vals:
                if distances[cluster[k]][key] <= r:
                    L_t_vals[key] = L_t_vals[key] - (r - distances[cluster[k]][key])
    t1 = time.time()
    clock = t1 - t0
    return clustering, clock

#input: positive adjacency matrix, clustering (as a list of lists), vector of positive degrees, p in lp norm
#output: vector of disagreements, objective value (max disagreements at a vertex), maximizing vertex
def LocalObj(S, clustering, pos_degrees, norm):
    #create positive adjacency matrix
    pos_adj_mx = positive_adjacency_from_signed(S)
    n = len(pos_degrees)
    disag_vector = np.zeros(n)
    num_clusters = len(clustering)
    for i in range(num_clusters):
        clus = clustering[i]
        for j in range(len(clus)):
            pos_disag = pos_degrees[clus[j]]
            neg_disag = 0
            for k in range(len(clus)):
                #positive edges inside the cluster are not disagreements
                if pos_adj_mx[clus[j]][clus[k]] == 1:
                    pos_disag = pos_disag - 1
                #negative edges inside the cluster are disagreements
                elif S[clus[j]][clus[k]] == -1:
                    neg_disag = neg_disag + 1
                #deleted edges contribute nothing
            disag_vector[clus[j]] = pos_disag + neg_disag
    alg_obj_val = np.linalg.norm(disag_vector, norm)

    if norm == math.inf:
        obj_vx = np.argmax(disag_vector)
    else:
        obj_vx = math.inf
    return disag_vector, alg_obj_val, obj_vx

#input: positive adjacency matrix
#output: vector of positive degrees
def DegreeDist(S):
    #create positive adjacency matrix
    pos_adj_mx = positive_adjacency_from_signed(S)
    n = np.shape(pos_adj_mx)[0]
    degrees = np.dot(pos_adj_mx, np.ones(n))
    return degrees

#input: signed matrix, LP solver (in Gurobi)
#output: LP objective value, LP solution
def MinMaxLPonly(S, method):
    n = np.shape(S)[0]
    upper_bounds = np.ones(int(n*(n-1)/2)+1)
    upper_bounds[int(n*(n-1)/2)] = grb.GRB.INFINITY
    M = grb.Model('my_model')
    K = []
    for i in range(n):
        for j in range(n-i-1):
            K.append((i,i+j+1))
    K.append((0,0))
    l = grb.tuplelist(K)
    x = M.addVars(l, name='x', ub=upper_bounds)
    for i in range(n-2):
        for j in range(n-2-i):
            for k in range(n-2-i-j):
                u = i
                v = i + j + 1
                w = i + j + k + 2
                M.addConstr(x[u,v] + x[v,w] >= x[u,w])
                M.addConstr(x[u,v] + x[u,w] >= x[v,w])
                M.addConstr(x[v,w] + x[u,w] >= x[u,v])
    add_min_max_vertex_constraints(M, x, S)
    M.setObjective(x[0,0], grb.GRB.MINIMIZE)
    M.setParam('Method', method)
    M.optimize()
    distances = np.zeros([n,n])
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            distances[u][v] = x[u,v].x
            distances[v][u] = x[u,v].x
    return M.objVal, distances
#input: number of vertices, distances, radii r and r2 for rounding algorithm
#output: L_0 values, R and R2 neighborhoods
def MinMaxLPneighbors(n, distances, r, r2):
    #initialize a dictionary that stores the L_t values of each vertex
    L_t_vals = {}
    #initialize the dictionaries that store the r and r2 neighborhoods of each vertex
    neighborsR = {}
    neighborsR2 = {}
    for k in range(n):
        L_t_vals.update({k: 0})
        neighborsR.update({k: []})
        neighborsR2.update({k: []})

    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            #add to r2-neighborhood
            if distances[u][v] <= r2:
                neighborsR2[u].append(v)
                neighborsR2[v].append(u)
                #update L_t values and add to r-neighborhood
                if distances[u][v] <= r:
                    L_t_vals[u] = L_t_vals[u] + r - distances[u][v]
                    neighborsR[u].append(v)
                    L_t_vals[v] = L_t_vals[v] + r - distances[u][v]
                    neighborsR[v].append(u)
    return L_t_vals, neighborsR, neighborsR2


#Combines MinMaxLPonly and MinMaxLPneighbors
def MinMaxLP(S, r, r2, method):
    t0 = time.time()
    n = np.shape(S)[0]
    upper_bounds = np.ones(int(n*(n-1)/2)+1)
    upper_bounds[int(n*(n-1)/2)] = grb.GRB.INFINITY
    M = grb.Model('my_model')
    K = []
    for i in range(n):
        for j in range(n-i-1):
            K.append((i,i+j+1))
    K.append((0,0))
    l = grb.tuplelist(K)
    x = M.addVars(l, name='x', ub=upper_bounds)
    for i in range(n-2):
        for j in range(n-2-i):
            for k in range(n-2-i-j):
                u = i
                v = i + j + 1
                w = i + j + k + 2
                M.addConstr(x[u,v] + x[v,w] >= x[u,w])
                M.addConstr(x[u,v] + x[u,w] >= x[v,w])
                M.addConstr(x[v,w] + x[u,w] >= x[u,v])
    add_min_max_vertex_constraints(M, x, S)
    M.setObjective(x[0,0], grb.GRB.MINIMIZE)
    M.setParam('Method', method)
    M.setParam("Crossover", 0)
    M.optimize()
    L_t_vals = {}
    neighborsR = {}
    neighborsR2 = {}
    for k in range(n):
        L_t_vals.update({k: 0})
        neighborsR.update({k: []})
        neighborsR2.update({k: []})
    distances = np.zeros([n,n])
    for u in range(n):
        for w in range(n-u-1):
            v = u + w + 1
            distances[u][v] = x[u,v].x
            distances[v][u] = x[u,v].x
            if distances[u][v] <= r2:
                neighborsR2[u].append(v)
                neighborsR2[v].append(u)
                if distances[u][v] <= r:
                    L_t_vals[u] = L_t_vals[u] + r - distances[u][v]
                    neighborsR[u].append(v)
                    L_t_vals[v] = L_t_vals[v] + r - distances[u][v]
                    neighborsR[v].append(u)
    t1 = time.time()
    clock = t1 - t0
    return M.objVal,distances,L_t_vals, neighborsR,neighborsR2,clock
