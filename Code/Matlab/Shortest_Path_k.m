function [SP_k, Path] = Shortest_Path_k(weights, k)
    %input the graph with the number of clusters, outpur the total cost,
    %and Path
    [n,~] = size(weights);
    SP = Inf*ones(n,k);
    P = zeros(n,k);
    SP(:,1) =weights(1,:);
    for e = 2:k
        SP(:,e) = SP(:,e-1);
        for i = 1:n
            for j = 1:i
                if SP(i,e) > SP(j,e-1) + weights(j,i)
                    SP(i,e) = SP(j,e-1) + weights(j,i);
                    P(i,e) = j;
                end 
            end 
        end 
    end
    [SP_k,optimal_k] = min(SP(n,:));
    Path = zeros(optimal_k+1,1);
    for i = optimal_k+1:-1:1
        if i== optimal_k+1
            Path(i) = n;
        elseif i == 1
            Path(i) = 1;
        else
            Path(i) = P(int64(Path(i+1)),i);
        end     
    end 
end

