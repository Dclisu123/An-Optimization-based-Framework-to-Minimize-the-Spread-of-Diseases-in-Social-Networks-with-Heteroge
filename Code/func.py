# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:47:40 2020

@author: dclis
"""


import numpy as np
import copy
import cvxpy as cp
from scipy.linalg import sqrtm

def r_gen(n):
    # age_group, race_group = C.shape
    # age_population = age_pro*total_population
    # race_pro = race_population/sum(race_population)
    # risk = np.zeros((age_group,race_group))
    # for i in range(age_group):
    #     for j in range(race_group):
    #         risk[i,j] = C[i,j]/(age_population[i]*race_pro[j])
    # rand_age = np.random.choice(age_group,n,p = age_pro)
    # rand_race = np.random.choice(race_group,n,p = race_pro)
    pro, risk = np.genfromtxt('epidemic_parameter.csv',delimiter=',',usecols=(0,1),unpack=True)
    rand_demo = np.random.choice(len(pro),n,p = pro)
    r = np.zeros((n,1))
    age = np.zeros((n,1))
    race = np.zeros((n,1))
    d = np.zeros((n,1))
    for i in range(n):
        r[i,0] = risk[int(rand_demo[i])]
        age[i,0],race[i,0] = divmod(rand_demo[i],6)        
        d[i,0] = float(np.random.uniform(0,1,1))
    return r,age,race,d



def r_gen_new(n):
    pro = [0.004591,0.009578754,0.00274937,0.005736339,0.098066036,0.204606914,0.120510269,0.251435006,0.002478649,0.003909622,0,0,0.048210531,0.076043415,0.016341904,0.025776405,0.000401334,0.000945424,0,0,0.031441269,0.074066346,0.006887201,0.016224212]
    risk = [0.004417,0.003650413,0.017667999,0.014601652,0.001369898,0.001132147,0.005479593,0.00452859,0.011441239,0.00945557,0.045764957,0.037822278,0.00252041,0.002082984,0.01008164,0.008331934,0.008207378,0.006782957,0.03282951,0.027131827,0.002066937,0.001708212,0.008267748,0.00683285]
    rand_demo = np.random.choice(len(pro),n,p = pro)
    r = np.zeros((n,1))
    age = np.zeros((n,1))
    health = np.zeros((n,1))
    vacc = np.zeros((n,1))
    obe = np.zeros((n,1))
    for i in range(n):
        r[i,0] = risk[int(rand_demo[i])]
        age[i,0],rem = divmod(rand_demo[i],8)    
        health[i,0],rem = divmod(rem,4)      
        vacc[i,0],rem = divmod(rem,2) 
        obe[i,0],_  = divmod(rem,1) 
    return r,age,health,vacc,obe
    
    
def obj_cal(data_cluster,sigmamu):
   length = len(data_cluster)
   mul = 1
   summation = 0
   for i in range(length):
       mul = mul*(1-sigmamu*data_cluster[i])
       summation = summation+(1-data_cluster[i])/(1-sigmamu*data_cluster[i])
   exp = length - mul*summation
   edge_cost = (length)**2/2
   return exp,edge_cost
   

def eta_estimate(Gdp_capital,avg_cost,loss_low,loss_high,n):
    eta_low = (2*loss_low*Gdp_capital)/((n-1)*avg_cost)
    eta_high = (2*loss_high*Gdp_capital)/((n-1)*avg_cost)
    return eta_low,eta_high

#def shortest_path_k(G,k):
    
def weights_cal(data,sigmamu,eta):    
    n = len(data)
    weights = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range (i+1,n+1):
            exp,edge_cost = obj_cal(data[i:j],sigmamu)
            weights[i,j] = exp - eta*edge_cost
    return weights



def weights_matrix_cal(data,sigmamu,eta):
    n = len(data)
    r1=[(1-data[i])/(1-sigmamu*data[i]) for i in range(n)]
    r2=[np.log(1-sigmamu*data[i]) for i in range(n)]
    matrix_weights = np.asarray([[j-i-sum(r1[i:j])*np.exp(sum(r2[i:j]))-eta * (j-i)**2/2 if j>i else 0.0 for j in range(n+1)] for i in range(n+1)])
    return matrix_weights



def calculate(data,sigmamu,eta,k):
    # start_time = time.time()
    weights = weights_matrix_cal(data,sigmamu,eta)
    # weights = np.zeros((n+1,n+1))
    # for i in range(n+1):
    #     for j in range (i+1,n+1):
    #         exp,edge_cost = obj_cal(data[i:j],sigmamu)
    #         weights[i,j] = exp - eta * edge_cost 
    # cal_edge_time = time.time()- start_time
    # print('weight calculation time: %.4f' %cal_edge_time)
    SP_k,path,optimal_k = shortest_k_new(weights,k)
    # print('new method obj:%.4f , %.4f' %(SP_k,optimal_k))
    # print("The path is: ", path)
    # SP_k,path,optimal_k = shortest_k(weights,k)
    # print('old method obj:%.4f , %.4f' %(SP_k,optimal_k))
    # print("The path is: ", path)
    # solve_time = time.time() - start_time - cal_edge_time
    # print('shortest path solve time: %.4f' %solve_time)
    return SP_k,path,optimal_k


def shortest_k(weights,k):
    n,_ = weights.shape
    count = 1
    SP = float('inf')*np.ones((n,n,k+1))
    P = np.zeros((n,n,k+1))
    for e in range(k+1):
        for i in range(n):
            for j in range(n):
                SP[i,j,e] = float('inf')
                if e==0 and i==j:
                    SP[i,j,e] = 0
                if e==1 and weights[i,j] != float('inf'):
                    SP[i,j,e] = weights[i,j]
                if e>1:
                    for a in range(n):
                        if weights[i,a] != float('inf') and i != a and j != a and SP[a,j,e-1] != float('inf'):
                            if SP[i,j,e] > weights[i,a] + SP[a,j,e-1]:
                                SP[i,j,e] = weights[i,a] + SP[a,j,e-1]
                                P[i,j,e] = a
    min_SP = float('inf')
    for i in range(k):
        if min_SP>SP[0,n-1,i+1]:
            min_SP = SP[0,n-1,i+1]
            optimal_k = i+1
    Path = np.zeros((optimal_k+1))
    Path[0] = 0
    Path[1] = P[0,n-1,optimal_k]
    for e in range(optimal_k,2,-1):
        count = count+1
        Path[count] = P[int(Path[count-1]),n-1,e-1]
        
    Path[optimal_k] = n-1
    
    SP_k = SP[0,n-1,optimal_k]
    
    return SP_k,Path,optimal_k
    



def shortest_k_new(weights,k):
    n,_ = weights.shape
    SP = float('inf')*np.ones((n,k))
    P = np.zeros((n,k))
    SP[:,0] = weights[0,:]
    optimal_k = 0
    for e in range(1,k):
        cur_k_optimal = float('inf')*np.ones(n)
        for i in range(n):
            for j in range(i):
                if cur_k_optimal[i] > SP[j,e-1] + weights[j,i]:
                    cur_k_optimal[i] = SP[j,e-1] + weights[j,i]
                    P[i,e] = j
            if cur_k_optimal[i] < SP[i,e-1]:
                SP[i,e] = cur_k_optimal[i]
            else:
                SP[i,e] = SP[i,e-1]
    SP_k = min(SP[n-1,:])
    optimal_k = np.argmin(SP[n-1,:])+1
    Path = np.zeros((optimal_k+1))
    for i in range(optimal_k,-1,-1):
        if i == optimal_k:
            Path[i] = n-1
        elif i == 0:
            Path[i] = 0
        else:
            Path[i] = P[int(Path[i+1]),i]
    return SP_k,Path,optimal_k


def shortest_k_new_part(weights,obj_i,obj_j):
    n,_ = weights.shape
    k = obj_i
    SP_i= float('inf')*np.ones((obj_i+1,k))
    optimal_k_i = np.zeros((obj_i+1))
    P_i = np.zeros((obj_i+1,k))
    SP_i[:,0] = weights[0,:obj_i+1]
    for e in range(1,k):
        cur_k_optimal = float('inf')*np.ones(obj_i+1)
        for i in range(obj_i+1):
            for j in range(i):
                if cur_k_optimal[i] > SP_i[j,e-1] + weights[j,i]:
                    cur_k_optimal[i] = SP_i[j,e-1] + weights[j,i]
                    P_i[i,e] = j
            if cur_k_optimal[i] < SP_i[i,e-1]:
                SP_i[i,e] = cur_k_optimal[i]
            else:
                SP_i[i,e] = SP_i[i,e-1]
    for i in range(obj_i+1):
        optimal_k_i[i] = np.argmin(SP_i[i,:])+1    
    Path_list_i = []
    for i in range(obj_i+1):
        Path = np.zeros((int(optimal_k_i[i]+1)))
        for j in range(int(optimal_k_i[i]),-1,-1):
            if j == optimal_k_i[i]:
                Path[j] = i
            elif j == 0:
                Path[j] = 0
            else:
                Path[j] = P_i[int(Path[j+1]),j]
        Path_list_i.append(Path)

       
                
    k = (n-obj_j)-1
    SP_j= float('inf')*np.ones((n-obj_j,k))
    P_j = np.zeros((n-obj_j,k))
    SP_j[:,0] = np.flip(weights[obj_j:n,n-1])
    optimal_k_j = np.zeros((n-obj_j))
    for e in range(1,k):
        cur_k_optimal = float('inf')*np.ones(n-obj_j)
        for i in range(n-obj_j):
            for j in range(i):
                if cur_k_optimal[i] > SP_j[j,e-1] + weights[n-1-i,n-1-j]:
                    cur_k_optimal[i] = SP_j[j,e-1] + weights[n-1-i,n-1-j]
                    P_j[i,e] = j
            if cur_k_optimal[i] < SP_j[i,e-1]:
                SP_j[i,e] = cur_k_optimal[i]
            else:
                SP_j[i,e] = SP_j[i,e-1]
    for i in range(n-obj_j):
        optimal_k_j[i] = np.argmin(SP_j[i,:])+1   
    Path_list_j = []
    for i in range(n-obj_j):
        Path = np.zeros((int(optimal_k_j[i]+1)))
        for j in range(int(optimal_k_j[i]),-1,-1):
            if j == optimal_k_j[i]:
                Path[j] = i
            elif j == 0:
                Path[j] = 0
            else:
                Path[j] = P_j[int(Path[j+1]),j]
        Path = n-1-Path
        Path_list_j.append(Path)
                
                
                
    return SP_i[:,-1],SP_j[:,-1],Path_list_i,Path_list_j


def findsigmamu(r_vector,R0,epsilon):
    n= len(r_vector)
    I = n*(1-1/float(R0))
    a = 0.0
    b = 1.0
    mid = (a+b)/2.0
    i=0
    assert not diff(r_vector,a,I)* diff(r_vector,b,I)>0
    while(np.abs(diff(r_vector,mid,I))> epsilon ):
        if diff(r_vector,a,I)*diff(r_vector,mid,I)>0:
            a = mid
        else:
            b = mid
        i = i+1
        mid = (a+b)/2.0
    return mid
            
        
        

    
    
def diff(r_vector,sigmamu,I):
    exp,_ = obj_cal(r_vector, sigmamu)
    diff = exp - I
    return diff
    
def exp_r_vector(n,N):
    r_vector = np.zeros((N,n,2))
    for i in range(N):
        r,age,_,_ = r_gen(n)
        r_vector[i] =  np.column_stack((r,age))
        r_vector[i] = r_vector[i,r_vector[i,:,0].argsort(),:]
    mean_r = np.mean(r_vector,axis = 0)
    mean_r[:,1] = np.rint(mean_r[:,1])
    mean_rvec = mean_r[:,0].reshape((-1,1))
    return mean_rvec



def avg_time_infected(t,I,max_time,ini_inf):
    I_avg = np.zeros((max_time))
    t = np.ceil(t)
    for i in range(max_time):
        if(I[np.where(t==i)].size == 0):
            I_avg[i] = I_avg[i-1]
        else:
            I_avg[i] = np.mean(I[np.where(t==i)])
    return I_avg



def r_d_order(r_vector,d_vector,lam,growth_rate):
    n,_ = np.shape(r_vector)
    c_0 = np.ones((n,1))
    c_inf = np.divide(d_vector,r_vector.reshape(n,1))
    c = np.zeros((n,1))
    order_vector = np.zeros((n,1))
    for i in range(n):
        c[i] = c_0[i]*((1+np.exp(-growth_rate))/(1+np.exp(growth_rate*(lam-1)))) + c_inf[i]*(1-(1+np.exp(-growth_rate))/(1+np.exp(growth_rate*(lam-1))))
        order_vector[i] = r_vector[i]*c[i]
    order_ind = order_vector[:,0].argsort()
    return order_ind


def r_d_cal(r_vector,d_vector,lam,growth_rate,sigmamu,eta,k):
    order_ind = r_d_order(r_vector,d_vector,lam,growth_rate)
    r_vector = r_vector[order_ind]
    d_vector = d_vector[order_ind]
    n,_ = np.shape(r_vector)
    weights = float('inf')*np.ones((n+1,n+1))
    for i in range(n+1):
        for j in range (i+1,n+1):
            exp,edge_cost = obj_cal(r_vector[i:j-1],sigmamu)
            if i==j-1:
                k_means = 0
            else:
                k_means = np.sum(np.square((d_vector[i:j-1] - np.mean(d_vector[i:j-1]))))
            weights[i,j] = exp - eta * edge_cost + lam * growth_rate * k_means
    SP_k,path,optimal_k = shortest_k_new(weights,k)
    path = path.astype(int)
    Obj_exp = 0
    Obj_ecost = 0
    Obj_k_means = 0
    for i in range(len(path)-1):
        exp,edge_cost = obj_cal(r_vector[path[i]:path[i+1]],sigmamu)
        k_means = np.sum(np.square((d_vector[path[i]:path[i+1]] - np.mean(d_vector[path[i]:path[i+1]]))))
        Obj_exp = exp + Obj_exp
        Obj_ecost = edge_cost + Obj_ecost
        Obj_k_means =  k_means + Obj_k_means
    Obj_epi = Obj_exp + eta * (n**2/2 - Obj_ecost)
    return Obj_epi, Obj_exp,Obj_ecost,Obj_k_means



def generate_exp_r_vector(exp_num_test,N):
    exp_vector = np.zeros((N,1))
    for _ in range(exp_num_test):
        r_vector,_,_,_,_ = r_gen_new (N)
        r_vector[:,0].sort()
        exp_vector = exp_vector + r_vector
    exp_vector = exp_vector/exp_num_test
    return exp_vector 

def generate_ordering(N,T,G,r_vector,beta,gamma,K):
    ind = np.arange(N)
    order = []
    th_ind = np.linspace(0,(K-1)/K, num=K)
    while len(ind)>= K:
        cur_ind = (th_ind*(len(ind)-1)).astype(int)
        for k in range(K):
            order.append(ind[cur_ind[k]])   
        ind = np.delete(ind, cur_ind)
    for i in range(len(ind)):
        order.append(ind[i]) 
    order = list(dict.fromkeys(order))    
        
    clu = [[] for _ in range(K)]
    
    for i in range(len(order)):
        if i<=k:
            clu[i].append(order[i])
        else:
            val = np.zeros(K)
            for k in range(K):
                # G_sub_1 = G.subgraph(clu[k]+ [order[i]])
                # G_sub_2 = G.subgraph(clu[k])
                # val[k] = absolute_rank(G_sub_1) - absolute_rank(G_sub_2)
                for a,b in G.edges(order[i]):
                    if b in clu[k]:
                        val[k] = val[k] + abs(a-b)
            clu[np.argmin(val)].append(order[i])
    optimal_order = []
    for k in range(K):
        sorted_order = sorted(range(len(clu[k])), key=clu[k].__getitem__)
        clu[k] = [clu[k][i] for i in sorted_order]
        optimal_order = optimal_order + clu[k]

        
    return optimal_order


def generate_ordering_matrix(T,A,r_vector,beta,gamma,K):
    N = len(A)
    ind = np.arange(N)
    order = []
    th_ind = np.linspace(0,(K-1)/K, num=K)
    while len(ind)>= K:
        cur_ind = (th_ind*(len(ind)-1)).astype(int)
        for k in range(K):
            order.append(ind[cur_ind[k]])   
        ind = np.delete(ind, cur_ind)
    for i in range(len(ind)):
        order.append(ind[i]) 
    order = list(dict.fromkeys(order))    
        
    clu = [[] for _ in range(K)]
    
    for i in range(len(order)):
        if i < K:
            clu[i].append(order[i])
        else:
            val = np.zeros(K)
            for k in range(K):
                val[k] = A[order[i],clu[k]] @ abs(np.arange(N)[clu[k]] - order[i])
            clu[np.argmin(val)].append(order[i])
    optimal_order = []
    for k in range(K):
        sorted_order = sorted(range(len(clu[k])), key=clu[k].__getitem__)
        clu[k] = [clu[k][i] for i in sorted_order]
        optimal_order = optimal_order + clu[k]

        
    return optimal_order



def sum_weight_path(weights,path):
    obj = 0
    for i in range(len(path)-1):
        obj += weights[int(path[i]),int(path[i+1])]
    return obj

def expected_simu(T,G_sub,r_vector,gamma,beta):
    N_sub = G_sub.number_of_nodes()
    sol_r = np.zeros((T+1,N_sub))
    node_list = list(G_sub.nodes())
    for i in range(N_sub):
        sol_r[0,i] = G_sub.nodes[node_list[i]]['risk']
    for t in range(T):
        for i in range(N_sub):
            sol_r[t+1,i] = (1-gamma)*sol_r[t,i] + (1-sol_r[t,i])* (1-np.prod([1-beta*sol_r[t,node_list.index(n)] for n in G_sub[node_list[i]]]))
    cur_obj = np.sum(sol_r[1:,:])   
    return cur_obj   


def weights_matrix_cal_general(T,G,r_vector,beta,gamma,K):
    N = G.number_of_nodes()
    optimal_order = generate_ordering(N,T,G,r_vector,beta,gamma,K)
    matrix_weights = np.zeros((N+1,N+1))
    num_of_edges = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(i,N+1):
            G_sub = G.subgraph(optimal_order[i:j])
            matrix_weights[i,j] = expected_simu(T,G_sub,r_vector,gamma,beta)
            num_of_edges[i,j] = G_sub.number_of_edges()
    return matrix_weights,num_of_edges


def expected_simu_matrix(T,A_sub,r_vector_sub,gamma,beta):
    N_sub = len(A_sub)
    r_obj = np.zeros((N_sub,1))
    for t in range(T):
        prod = np.prod(1-beta*np.multiply(r_vector_sub,A_sub),axis = 0)
        r_vector_sub = (1-gamma)*r_vector_sub + np.multiply((1-r_vector_sub),(1-prod).reshape(-1,1))
        r_obj = r_obj + r_vector_sub
    return np.sum(r_obj)/T
        

def weights_matrix_cal_general_matrix(T,A,r_vector,beta,gamma,K):
    optimal_order = generate_ordering_matrix(T,A,r_vector,beta,gamma,K)
    N = len(A)
    matrix_weights = np.zeros((N+1,N+1))
    num_of_edges = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(i+1,N+1):
            matrix_weights[i,j] = expected_simu_matrix(T,A[:,optimal_order[i:j]][optimal_order[i:j],:],r_vector[optimal_order[i:j]],gamma,beta)
            num_of_edges[i,j] = num_of_edges[i,j-1] + np.sum(A[optimal_order[j-1],optimal_order[i:j]])
    return matrix_weights,num_of_edges,optimal_order


def calculate_general(T,G,r_vector,beta,gamma,eta,K):
    weights,_ = weights_matrix_cal_general(T,G,r_vector,beta,gamma,eta,K)
    SP_k,path,optimal_k = shortest_k_new(weights,K)
    SP_k = SP_k + eta * G.number_of_edges()
    return SP_k,path,optimal_k  




def calculate_general_matrix(T,A,r_vector,beta,gamma,eta,K):
    weights,edges,optimal_order = weights_matrix_cal_general_matrix(T,A,r_vector,beta,gamma,K)
    new_weight = weights - eta * edges
    SP_k,path,optimal_k = shortest_k_new(new_weight,K)
    optimal_result = []
    for i in range(optimal_k):
        clu = optimal_order[int(path[i]):int(path[i+1])]
        optimal_result.append(clu)      
    
    return SP_k,optimal_result,optimal_k 


def max_k_cut(T,A,K,r_vector,beta,gamma,eta):
    ori_expected = expected_simu_matrix(T,A,r_vector,gamma,beta)
    N = len(A)
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            if A[i,j] == 1:
                A_sub = copy.deepcopy(A)
                A_sub[i,j] = 0
                A_sub[j,i] = 0
                W[i,j] = ori_expected - expected_simu_matrix(T,A_sub,r_vector,gamma,beta) - eta

    
    X = cp.Variable((N,N), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        X[i,i] == 1 for i in range(N)
    ]
    for i in range(N):
        for j in range(N):
            if i != j:
                constraints += [
                    X[i,j] >= -1/(K-1)
                ]
    
    
    objective = (K-1)/K*cp.sum(cp.multiply(W,np.ones((N,N)) - X))
    
    prob = cp.Problem(cp.Maximize(objective),constraints)
    prob.solve()
    
    z_vector = np.random.standard_normal(size=(N,K))
    
    x = sqrtm(X.value)
    
    max_k_partition = [[] for _ in range(K)]
    for i in range(N):
        val = np.zeros((K))
        for k in range(K):
            val[k] = x[i,:]@z_vector[:,k]
        max_k_partition[np.argmax(val)].append(i)
    
    max_k_obj = 0
    for k in range(K):
        max_k_obj = max_k_obj + expected_simu_matrix(T,A[:,max_k_partition[k]][max_k_partition[k],:],r_vector[max_k_partition[k]],gamma,beta) - eta*np.sum(A[:,max_k_partition[k]][max_k_partition[k],:])/2
        
    max_k_obj = max_k_obj + eta*np.sum(A)/2
    
    return max_k_obj,max_k_partition

        

    