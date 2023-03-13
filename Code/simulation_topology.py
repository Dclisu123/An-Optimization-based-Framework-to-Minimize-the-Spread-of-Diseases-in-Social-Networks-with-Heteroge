# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:17:00 2022

@author: dclisu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:20:59 2022

@author: dclisu
"""
import numpy as np
import networkx as nx
from func import r_gen_new,calculate_general_matrix
import copy
import scipy.stats as st
import matplotlib.pyplot as plt
import concurrent.futures
import time
import EoN
import argparse
## simulation for four different topologies


def run_simu():
    
    number_of_edges = np.zeros(5)
    
    r_vector,age_vector,health_vector,vacc_vector,obe_vector = r_gen_new (N)
    ind = r_vector[:,0].argsort()

    r_vector_sorted = r_vector[ind]
    
    K = 2

    #solution to the clustering problem --- based on K=2
    SP_k,optimal_result,optimal_k = calculate_general_matrix(T,A,r_vector_sorted,beta,gamma,eta,K)
    
    health_result = []   
    for i in range(K):
        health_result.append(np.where(health_vector==i)[0])
    
    vacc_result = []
    for i in range(K):
        vacc_result.append(np.where(vacc_vector==i)[0])
        
    obe_result = []
    for i in range(K):
        obe_result.append(np.where(obe_vector==i)[0])
        

        
    ini_set = []
    while(len(ini_set)<1):
        for i in range(N):
            if(np.random.binomial(1, p=r_vector[i])==1):
                ini_set.append(i)
 
    
    
    #Orginal graph
    t_ori, _, I_ori = EoN.fast_SIS(G, beta, gamma, tmax = max_time,initial_infecteds = ini_set)   
    I_avg_ori = EoN.subsample(np.linspace(0,max_time,time_int),t_ori,I_ori)
    number_of_edges[0] = G.number_of_edges()
    
    #Optimal partition
    G_optimal = copy.deepcopy(G)
    for cur_arr in optimal_result :
        for i_node in cur_arr:
            for j_node in np.setdiff1d(np.arange(N),cur_arr):
                if G_optimal.has_edge(int(i_node), j_node):
                    G_optimal.remove_edge(int(i_node),j_node)
                    
    t_optimal, _, I_optimal = EoN.fast_SIS(G_optimal , beta, gamma, tmax = max_time ,initial_infecteds = ini_set)
    # I_avg_optimal_age = avg_time_infected(t_optimal_age, I_optimal_age, max_time, len(ini_set))
    I_avg_optimal = EoN.subsample(np.linspace(0,max_time,time_int),t_optimal,I_optimal)
    number_of_edges[1] = G_optimal.number_of_edges()
    
    
    
    
    
    
    
    #Health partition
    G_health = copy.deepcopy(G)
    for cur_arr in health_result:
        for i_node in cur_arr:
            for j_node in np.setdiff1d(np.arange(N),cur_arr):
                if G_health.has_edge(int(i_node), j_node):
                    G_health.remove_edge(int(i_node),j_node)    
    t_health, _, I_health = EoN.fast_SIS(G_health, beta, gamma, tmax = max_time ,initial_infecteds = ini_set)
    # I_avg_age = avg_time_infected(t_age, I_age, max_time, len(ini_set))
    I_avg_health = EoN.subsample(np.linspace(0,max_time,time_int),t_health,I_health)
    number_of_edges[2] = G_health.number_of_edges()
    
    


    G_vacc = copy.deepcopy(G)
    for cur_arr in vacc_result:
       for i_node in cur_arr:
           for j_node in np.setdiff1d(np.arange(N),cur_arr):
               if G_vacc.has_edge(int(i_node), j_node):
                   G_vacc.remove_edge(int(i_node),j_node)     
    t_vacc, _, I_vacc = EoN.fast_SIS(G_vacc, beta, gamma, tmax = max_time ,initial_infecteds = ini_set)
    # I_avg_race = avg_time_infected(t_race, I_race, max_time, len(ini_set))
    I_avg_vacc = EoN.subsample(np.linspace(0,max_time,time_int),t_vacc,I_vacc)
    number_of_edges[3] = G_vacc.number_of_edges()
       

    G_obe = copy.deepcopy(G)
    for cur_arr in obe_result:
       for i_node in cur_arr:
           for j_node in np.setdiff1d(np.arange(N),cur_arr):
               if G_obe.has_edge(int(i_node), j_node):
                   G_obe.remove_edge(int(i_node),j_node)     
    t_obe, _, I_obe = EoN.fast_SIS(G_obe, beta, gamma, tmax = max_time ,initial_infecteds = ini_set)
    # I_avg_race = avg_time_infected(t_race, I_race, max_time, len(ini_set))
    I_avg_obe = EoN.subsample(np.linspace(0,max_time,time_int),t_obe,I_obe)
    number_of_edges[4] = G_obe.number_of_edges()
    


    result = np.asarray([I_avg_ori,I_avg_optimal,I_avg_health,I_avg_vacc,I_avg_obe])
    partition = [optimal_result,health_result,vacc_result,obe_result]
    
    return result,number_of_edges,partition
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-g', type=int, required=True)
    parser.add_argument('-r0', type = int, required=True)
    # Parse the argument
    args = parser.parse_args()
    
    
    N = 500
    T = 30
    avg_degree = 10
    R0_type= args.r0
    graph_type = args.g
    Gdp_capital = 65118.4
    cost_low = 13297*1.2
    cost_high = 40218*1.2
    avg_cost = (cost_low+cost_high)/2.0
    gamma = 1/11
    loss = 0
    num_exp = 100
    max_time = 300
    time_int = 300
    
    if R0_type ==0:
        R0 = 3.8
    elif R0_type ==1:
        R0 = 5.7
    elif R0_type ==2:
        R0 = 8.9
    else:
        R0 = 3.8
    
    
    
    if graph_type == 0:
        G = nx.nx.complete_graph(N)
    elif graph_type ==1:
        G = nx.expected_degree_graph([avg_degree for i in range(N)],selfloops = False)
    elif graph_type == 2:
        G = nx.connected_caveman_graph(25, 20)
    elif graph_type ==3:
        G= nx.windmill_graph(25, 21)
        N = 501
    else:
        G = nx.nx.complete_graph(N)
        
        
    degrees = [val for (node, val) in G.degree()]
    theta = (sum([i*i for i in degrees])/N)/(sum(degrees)/N)
    gamma = 1/11
    beta = R0*gamma/theta
    eta = N*loss*Gdp_capital/(G.number_of_edges()*avg_cost)
    A = nx.to_numpy_array(G)
         
    simulation_result = []
    simulation_edges = []
    partition_result = []
    start_time = time.time()
    # simulation_result = [run_simulation() for _ in range(num_exp)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run_simu) for _ in range(num_exp)] 
    for lines in concurrent.futures.as_completed(results):
        result, number_of_edges,partition =lines.result()
        simulation_result.append(result)
        simulation_edges.append(number_of_edges)
        partition_result.append(partition)
    np.savez('results_r0_'+str(R0)+'_graph_'+str(graph_type)+'.npz', simulation_result = simulation_result, simulation_edges = simulation_edges)
    total_time = time.time() - start_time
    
    
    print('finish time:%.4f' %total_time)
    
    
    
    ######## Exp finishes, read ###########
    
    time_int = 300

    results = np.load('results_r0_'+str(R0)+'_graph_'+str(graph_type)+'.npz')
    simulation = results['simulation_result']
    edge = results['simulation_edges']
    simulation_mean = np.mean(simulation,axis = 0)
    lower,upper = st.norm.interval(alpha=0.95, loc=np.mean(simulation,axis = 0 ), scale=st.sem(simulation,axis = 0))
    fig, ax = plt.subplots()
    p1 = ax.plot(range(time_int), simulation_mean[1,:],'k-' , linewidth=1 ,label = 'ISSE')
    ax.fill_between(range(time_int), lower[1,:], upper[1,:], color='k', alpha=.2,label = '95% confidence interval')
    p2 = ax.plot(range(time_int), simulation_mean[2,:],'k--' , linewidth=1 ,label = 'HB-P')
    ax.fill_between(range(time_int), lower[2,:], upper[2,:], color='k', alpha=.2)
    p3 = ax.plot(range(time_int), simulation_mean[3,:],'k-.' , linewidth=1 ,label = 'VB-P')
    ax.fill_between(range(time_int), lower[3,:], upper[3,:], color='k', alpha=.2)
    p4 = ax.plot(range(time_int), simulation_mean[4,:],'k:' , linewidth=1 ,label = 'OB-P')
    ax.fill_between(range(time_int), lower[4,:], upper[4,:], color='k', alpha=.2)
    a5 = ax.fill(np.NaN, np.NaN, facecolor = 'k',alpha = 0.2)
    
    plt.xlabel('Time (in days)')
    plt.ylabel('Number of infected subjects')
    #plt.legend(prop={'size': 10},frameon=False)
    plt.savefig('Simulation_r0_' + str(R0) + '_graph_' + str(graph_type)+'.pdf')
    plt.show()
                
                
    figlegend = plt.figure(figsize=(16,1))
    figlegend.legend([p1[0],p2[0],p3[0], p4[0], a5[0]], ['ISSE',"HB-P","VB-P","OB-P","95% Confidence Interval"],ncol=5,loc = "center",fontsize=16,frameon=False)
    figlegend.savefig('policy_legend.pdf')
    
    
    
    
    
    
    
    
    
    
    
    
 
    
    
