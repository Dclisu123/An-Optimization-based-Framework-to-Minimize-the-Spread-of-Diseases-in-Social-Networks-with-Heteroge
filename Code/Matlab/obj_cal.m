function [exp,edge_cost,obj]  = obj_cal(data_cluster, sigmamu,eta)
%% calculate the expcected value for clustering this list
   len = length(data_cluster);
   mul = 1;
   summation = 0;
   for i =1:len
       mul = mul*(1-sigmamu*data_cluster(i));
       summation = summation+(1-data_cluster(i))/(1-sigmamu*data_cluster(i));
   end 
   exp = len - mul*summation;
%% Calculate the edge cutting cost
    edge_cost = len^2/2;
%% Objective calculation
    obj = exp - eta*edge_cost;
%% count cluster sensitive attribtues
%    [~,m] = size(count_S);
%    count_clu = zeros(1,m);
%    for i = 0:m-1
%        count_clu(i+1)= sum(S_cluster(:)==i); 
%    end 
   
   
%% Calculate HGR
%   HGR = 0;
%   for i = 1:m
%       HGR = HGR + (count_clu(i)/n)^2/((len/n)*(count_S(i)/n));
%   end
      

end 

