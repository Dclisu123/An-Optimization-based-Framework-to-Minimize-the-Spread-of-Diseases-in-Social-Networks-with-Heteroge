# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:20:59 2022

@author: dclisu
"""
import numpy as np
import networkx as nx
from func import generate_exp_r_vector,weights_matrix_cal_general_matrix,shortest_k_new
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import concurrent.futures
import time
import argparse
## mu_sigma eta threshold:for each sigmamu, at which eta threshold should not qurantine


def th_1(beta):
    start = time.time()
    weights,num_of_edges,optimal_order = weights_matrix_cal_general_matrix(T,A,r_vector,beta,gamma,K)
    print('Part 1 - weight time:',time.time() - start)
    optimal_k = np.ones((len(eta_list)))*float('inf')
    
    th_eta_low = 0
    
    start = time.time()
    for j in range(len(eta_list)):
        new_weights = weights - eta_list[j]*num_of_edges 
        _,_,optimal_k[j] = shortest_k_new(new_weights,K)
        if optimal_k[j] == 1:
            break
        
    print('Bellman-ford time:',time.time() - start)
    
    print('finish',beta)    
    
    return th_eta_low,optimal_k



def import_npz(npz_file):
    Data = np.load(npz_file,allow_pickle = True)
    for varName in Data:
        globals()[varName] = Data[varName]
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-g', type=int, required=True,help = 'value for network type')
    # Parse the argument
    args = parser.parse_args()
    graph_type = args.g
    
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
    beta_C = np.array([3.8,8.9])*gamma/theta
    A = nx.to_numpy_array(G)
    print('Generation time:',time.time() - start)
    print('Generation done \n')
    
    ## mu_sigma eta threshold:for each mu_sigma, at which sigmamu threshold should not qurantine
    K = N
    b_interval = 109
    e_interval = 100
    beta_list = np.linspace(beta[0],beta[1],b_interval+1)
    beta_list = np.append(beta_list,beta_C)
    eta_list = np.linspace(eta[0],eta[1],e_interval+1)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(th_1,beta_list)
    eta_low_list = []
    optimal_len_list = []
    for lines in results:
        eta_low,optimal_len = lines
        eta_low_list.append(eta_low)
        optimal_len_list.append(optimal_len)
    print('3region done \n')
    np.savez('performance_'+ name + '.npz', eta_low_list = eta_low_list , optimal_len_list = optimal_len_list,beta_list_p1 = beta_list,eta_list_p1 = eta_list)

    

    ##################Exp finishes, npz #####################
    import_npz('performance_'+ name+ '.npz')
    
    ####################First graph#####################################
    beta_C = beta_list_p1[[-2,-1]]
    beta_list_p1 = beta_list_p1[:110]
    first_th = np.zeros(len(beta_list_p1))
    last_th = np.zeros(len(beta_list_p1))
    optimal_len_list[:,0] = N
    loss_list = np.linspace(0,0.45,len(eta_list_p1))
    for i in range(len(beta_list_p1)):
        for j in range(len(eta_list_p1)):
            if optimal_len_list [i,j] == N:
                first_th[i] = loss_list[j]
            if optimal_len_list [i,j] == float('inf'):
                break
        last_th[i] = loss_list[j-1]
        optimal_len_list[i,j:] = 1
    
    
    r_0_list = np.linspace(0,20,110)
    last_th[0] = 0
    fig, ax = plt.subplots()

    
    x_max = len(last_th)-1
    
    last_th = np.sort(last_th)
    ax.plot(last_th[0:x_max],r_0_list[0:x_max],color='k') ## Do not cluster
    xfill_2 = np.linspace(min(last_th),max(last_th),600)
    yfill_2 = np.interp(xfill_2, last_th, r_0_list)
    xfill_2 = xfill_2[:-1]
    yfill_2 = yfill_2[:-1]
    region_2 = ax.fill_between(xfill_2,yfill_2, interpolate=True, color = 'k', alpha=0.3, label='No restrictions')
    plt.ylim([min(r_0_list),max(r_0_list)])
    plt.locator_params(axis='x', nbins=6)
    ax.add_patch(Rectangle((0, 3.8), max(last_th), 8.9-3.8,ls = '--',fill=None))
    plt.xlabel('GDP Loss ($\mathcal{L})\%$',fontsize=14)
    xvals = ax.get_xticks()
    ax.set_xticklabels(["{:,.1%}".format(x) for x in xvals], fontsize=12)
    plt.ylabel('Basic reproduction number ($R_0$)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=False,loc = 'upper left',fontsize=12)
    plt.savefig('3_region_new_'+name+'.eps',format = 'eps')
    
    
    
    ###################Second graph################################
    low_num_clu_list = optimal_len_list[110,:]
    high_num_clu_list = optimal_len_list[111,:]
    for i in range(len(eta_list_p1)):
        if low_num_clu_list[i] == float('inf'):
            low_num_clu_list[i] = 1
        if high_num_clu_list[i] == float('inf'):
            high_num_clu_list[i] = 1
    
    x_max = len(high_num_clu_list)-1
    for i in range(len(high_num_clu_list)):
        if high_num_clu_list[i]==1:
            x_max = i
            break
    
    
    fig, ax = plt.subplots()
    ax.plot(loss_list,low_num_clu_list,'k-',label='$R_0$ =3.8') ## Always cluster individually
    ax.plot(loss_list,high_num_clu_list,'k--',label='$R_0$ =8.9') ## Always cluster individually
    
    plt.xlabel('GDP Loss ($\mathcal{L})\%$',fontsize=14)
    plt.ylabel('Number of social groups',fontsize=14)
    plt.locator_params(axis='x', nbins=6)
    plt.xlim([0, loss_list[13]])
    xvals = ax.get_xticks()
    ax.set_xticklabels(["{:,.1%}".format(x) for x in xvals], fontsize=12)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=False,loc = 1,fontsize=12)
    plt.savefig('num_of_cluster_new_'+name+'.eps',format = 'eps')
            

    
