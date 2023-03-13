function [sigmamu_low,sigmamu_mid,sigmamu_high,eta_low,eta_high] = para_cal(data,R0_low,R0_mid,R0_high,loss_low,loss_high,Gdp_capital,avg_cost)
    [n,~] = size(data);
    sigmamu_low = sigmamu_cal(data,R0_low);
    sigmamu_high = sigmamu_cal(data,R0_high);
    sigmamu_mid = sigmamu_cal(data,R0_mid);
    eta_low = (2*loss_low*Gdp_capital)/((n-1)*avg_cost);
    eta_high = (2*loss_high*Gdp_capital)/((n-1)*avg_cost);

end 