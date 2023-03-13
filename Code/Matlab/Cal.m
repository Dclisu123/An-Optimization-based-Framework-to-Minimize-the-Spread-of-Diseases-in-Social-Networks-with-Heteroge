function [Obj_Final,Obj_exp, Obj_ecost,Obj_obj, Obj_HGR, Path] = Cal(data, lambda, growth_rate,sigma,mu,eta,k)

     %[lambda_data,count_S,order] = pre(data,lambda,growth_rate);
     lambda_data = data;
%      lambda_n = lambda*growth_rate;
     [n,~] = size(data);
     %lambda_data = [(1:100)',(1:100)']

    
    
 %% use exact shortest path with k-edges
    weights = zeros(n+1,n+1);
    for i = 1:n+1
            for j = i+1:n+1
                [~,~,out_obj] = obj_cal(lambda_data(i:j-1,1), sigma*mu,eta);     
                weights(i,j) =  out_obj ;
            end
    end
    [SP_k, Path] = Shortest_Path_k(weights, k);   

         
    
    
    Obj_exp = 0;
    Obj_HGR = 0;
    Obj_ecost = 0;
    Obj_obj = 0;
    for i = 1:length(Path)-1
        [out_exp,out_ecost,out_obj] = obj_cal(lambda_data(Path(i):Path(i+1)-1,1), sigma*mu,eta);
        Obj_exp = Obj_exp + out_exp;
        Obj_ecost = Obj_ecost + out_ecost;
        Obj_obj = Obj_obj + out_obj;
%         Obj_HGR = Obj_HGR + out_HGR;
    end
%     Obj_HGR = Obj_HGR -1;
    Obj_ecost = -Obj_ecost+0.5*n^2;
    Obj_Final = Obj_obj ;
    
    
    
    
end