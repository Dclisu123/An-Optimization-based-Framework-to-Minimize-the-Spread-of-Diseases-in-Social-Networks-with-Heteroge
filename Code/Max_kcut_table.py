# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:17:32 2022

@author: dclisu
"""


import numpy as np
import networkx as nx
from func import max_k_cut,generate_exp_r_vector,weights_matrix_cal_general_matrix,shortest_k_new
import concurrent.futures
import time
import argparse
import pandas as pd





def evaluate(para_set):
    beta,eta,K = para_set
    
    start = time.time()
    weights,num_of_edges,_ = weights_matrix_cal_general_matrix(T,A,r_vector,beta,gamma,K)
    path_list = []
    new_weights = weights - eta*num_of_edges
    SP_k,cur_p,_ = shortest_k_new(new_weights,K)
    SP_k = SP_k + G.number_of_edges()*eta
    path_list.append(cur_p)
    print('shortest_path:',time.time()-start)
    
    start = time.time()
    max_k_obj = 0
    max_k_obj,max_k_partition = max_k_cut(T,A,K,r_vector,beta,gamma,eta)
    print('max-k-cut',time.time()-start)
    
    return SP_k,path_list,max_k_obj,max_k_partition

def import_npz(npz_file):
    Data = np.load(npz_file,allow_pickle = True)
    for varName in Data:
        globals()[varName] = Data[varName]

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument("--g",type=int,required=True, help="value for network type")
    parser.add_argument('--M', type=int, required=True,help="value for M")
    # Parse the argument
    args = parser.parse_args()
    
    
    exp_num_test = 10000
    N = 500
    T = 30
    R0 = np.array([0,20])
    loss = np.array([0,0.45])
    Gdp_capital = 65118.4
    cost_low = 13297*1.2
    cost_high = 40218*1.2
    avg_cost = (cost_low+cost_high)/2.0
    gamma = 1/11
    K = args.k
    graph_type = args.g
    
        
    start = time.time()
    avg_degree = 10
    
    if graph_type == 0:
        G = nx.nx.complete_graph(N)
        name = 'comp'
    elif graph_type ==1:
        G = nx.expected_degree_graph([avg_degree for i in range(N)],selfloops = False)
        name = 'expected'
    elif graph_type == 2:
        G = nx.connected_caveman_graph(25, 20)
        name = 'caveman'
    elif graph_type ==3:
        G= nx.windmill_graph(25, 21)
        name = 'windmill'
    else:
        G = nx.nx.complete_graph(N)
        name = 'comp'
        
        
    r_vector = generate_exp_r_vector(exp_num_test,N)
    degrees = [val for (node, val) in G.degree()]
    theta = (sum([i*i for i in degrees])/N)/(sum(degrees)/N)
    beta = R0*gamma/theta
    eta = N*loss*Gdp_capital/(G.number_of_edges()*avg_cost)
    A = nx.to_numpy_array(G) 
    print('Generation time:',time.time() - start)
    print('Generation done \n')
    
    
    # ### Random generated data and test on diffenret policies
    
    r_vector[:,0].sort()
    beta_list = np.array([3.8,5.7,8.9])*gamma/theta
    eta_list = N*np.array([0,0.15,0.45])*Gdp_capital/(G.number_of_edges()*avg_cost) 
    # ran_order = random.sample([i for i in range(N)],N)
    # P = np.zeros((N,N))
    # for i in range(N):
    #     P[ran_order[i],i] = 1
    # A = np.matmul(np.matmul(P,A_ori),np.transpose(P))
    para_set = []
    for beta in beta_list:
        for eta in eta_list:
                para_set.append((beta,eta,K))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(evaluate,para_set)
    SP_k_list = []
    path = []
    max_k_list = []
    max_k_partition_list = []
    for lines in results:
        SP_k,path_list,max_k_obj,max_k_partition = lines
        SP_k_list.append(SP_k)
        path.append(path_list)
        max_k_list.append(max_k_obj)
        max_k_partition_list.append(max_k_partition)
    print('Calculation done \n')
    np.savez('Max_kcut_table_graph_'+str(graph_type)+'_M_'+str(K)+'.npz', SP_k_list = SP_k_list,path = path, max_k_list = max_k_list,eta_list = eta_list,\
              r_vector_p3 = r_vector,max_k_partition_list = max_k_partition_list)
    
        
    ##################Exp finishes, read npz #####################
    import_npz('Max_kcut_table_graph_'+str(graph_type)+'_M_'+str(K)+'.npz')
    R0_list = np.array([[3.8]*3,[5.7]*3,[8.9]*3]).reshape(-1,1)
    L_list = np.array([0,0.15,0.45]*3).reshape(-1,1)
    SP_k_list = SP_k_list.reshape(-1,1)
    max_k_list = max_k_list.reshape(-1,1)
    df = pd.DataFrame(np.hstack((R0_list,L_list,SP_k_list,max_k_list,((max_k_list - SP_k_list)/SP_k_list))),columns = ['$R_0$','$\mathcal{L}$','ISSE','MC','Percentage reduction'])
    df.to_excel('Max_kcut_table_graph_'+str(graph_type)+'_M_'+str(K)+'.xlsx', index=False)

    

    
