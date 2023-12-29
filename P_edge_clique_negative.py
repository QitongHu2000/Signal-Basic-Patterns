# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.3
eta=0.3
t_0=100#10 #1000
n=50000000 #50000000 #1000000
B=0.1
alpha = 0.01

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def create_network_triangle(weight):
    G=nx.DiGraph()
    G.add_edge(0,1)
    G.add_edge(1,0)
    G.add_edge(1,2,weight=weight)
    G.add_edge(2,1,weight=weight)
    G.add_edge(0,2)
    G.add_edge(2,0)
    A=nx.to_numpy_matrix(G)
    return A

def create_network_edge(weight):
    G=nx.DiGraph()
    G.add_edge(0,1)
    G.add_edge(1,0)
    G.add_edge(1,2,weight=weight)
    G.add_edge(2,1,weight=weight)
    A=nx.to_numpy_matrix(G)
    return A

def simulation(A):
    def F(A,x,a,b):
        #B=0.01
        #alpha = 0.01
        return np.mat(-B*np.power(x,a)+alpha*A*np.power(x,b))
    
    def Fun(x,t,A,a,b):
        x=np.mat(x).T
        dx=F(A,x,a,b).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(x,t,A,a,b,source):
        x=np.mat(x).T
        dx=F(A,x,a,b).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def sim_first(a,b,A):
        x_0=np.ones(np.shape(A)[0])
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun,x_0,t,args=(A,a,b))
        x=xs[np.shape(xs)[0]-1,:].tolist()
        return x
    
    def sim_second(a,b,A,x,source):
        x[source]*=(1+gamma)
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun_1,x,t,args=(A,a,b,source))
        return np.mat(xs)
        
    def time(xs,eta):
        xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        for i in range(len(indexs)):
            len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
            len_2=eta-xs[indexs[i],i]
            times.append(indexs[i]+len_2/len_1)
        return np.mat(times)*t_0/n
    x=sim_first(a,b,A)
    xs=sim_second(a,b,A,x.copy(),source)
    times=time(xs.copy(),eta).tolist()[0]
    return times,x, xs

source=0
weights=np.logspace(3,5,10).astype('int')
a=1.2
b=1.0

times_simulation_edges=list()
for weight in weights:
    print(weight)
    A_edge=create_network_edge(weight)
    degrees=np.sum(A_edge,axis=1)
    times_simulation_edge,x_edge, xs_edge =simulation(A_edge)
    
    times_simulation_edge=times_simulation_edge[1]
    times_simulation_edges.append(times_simulation_edge)
    
times_simulation_edges=np.mat(times_simulation_edges)
save_dict(times_simulation_edges, 'P_edge_clique_P_'+str(int(100*a))+'_'+str(int(100*b)))
save_dict(weights, 'P_edge_clique_P_weight_'+str(int(100*a))+'_'+str(int(100*b)))

