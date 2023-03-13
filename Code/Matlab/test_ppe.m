n = 1000;
w=1;
R0_low = 3.8;
R0_mid = 5.7;
R0_high = 8.9;
loss_low = 0.15;
loss_high = 0.45;
Gdp_capital = 65118.4;
cost_low = 13297*1.2;
cost_high = 40218*1.2;
avg_cost = (cost_low+cost_high)/2.0;

data = Gen_risk_vacc(n);
data = sortrows(data,1);
total_edge = (n-1)*n/2;
total_path = {};

[sigmamu_low,sigmamu_mid,sigmamu_high,eta_low,eta_high] = para_cal(data(:,1),R0_low,R0_mid,R0_high,loss_low,loss_high,Gdp_capital,avg_cost);
eta_max = (2*Gdp_capital)/((n-1)*avg_cost);
% sigmamu_low= 0.11205863952636719;
% sigmamu_high = 0.18037033081054688;
% eta_low = 0.0035849226954524534;
% eta_high = 0.007681977204540972;

fprintf("Generation done \n");


%% mu_sigma eta threshold:for each eta, at which mu threshold should always quarantine individually
%change:musigma
k = n;
eta_more_ind = linspace(0,eta_max,101);
edge_cut_ratio = zeros(1,length(eta_more_ind));
parfor i = 1:length(eta_more_ind)
    [~, ~, e_cost,~,~,Path_B] = Cal(data, 0, 1 ,1,sigmamu_mid,eta_more_ind(i),n);
    edge_cut_ratio(i) = e_cost/total_edge;
    total_path{i} = Path_B;
end 

save('all_var_ppe_table.mat');


fprintf('Table done \n');


target = 261;
%target = 407;
cur_path = total_path{target};
clu = zeros(n,1);
for j =1: length(cur_path)-1
     clu(cur_path(j):cur_path(j+1)-1) = j;
end 

[uPhase,ia,ic] = unique(clu);
num_cluster = accumarray(ic, 1);
reduced_degrees = zeros(n,1);
for i =1: n
    reduced_degrees(i) = n- num_cluster(clu(i)) ;
end 


x = [815,185];
bar(x,'k');
set(gca,'xticklabel',{'185(Unprioritized)','815(Prioritezed)'});
ylabel('Number of subjects');
xlabel('Degree reduction');
box off

histogram(reduced_degrees);


[unique_degree,ia,ic] = unique(reduced_degrees);
reduced_degree_frequency = accumarray(ic, 1);

clu_degree = {};
for i = 1: length(unique_degree)
    clu_degree{i}=  data(find(reduced_degrees== unique_degree(i)),1:5);
end 

result = {};
for i = 1:length(unique_degree)
    for j = 1:4
        result{i,j,1} = length(clu_degree{i}(find(clu_degree{i}(:,j+1)==0)));
        result{i,j,2} = length(clu_degree{i}(find(clu_degree{i}(:,j+1)==1)));
        if j==1
            result{i,j,3} = length(clu_degree{i}(find(clu_degree{i}(:,j+1)==2)));
        end 
    end 
end 


attr = ["age","occupation status","vaccination status","obesity status"];
for j = 1:4
    figure
    y = [];
    legendInfo = {};
    if j == 1
        legendInfo{1} = ['[0,49]'];
        legendInfo{2} = ['[50,64]'];
        legendInfo{3} = ['>=65'];
        legendInfo{4} = ['Average' newline 'percentage'];
    elseif j==2
        legendInfo{1} = ['Health' newline 'worker'];
        legendInfo{2} = ['Non-health' newline 'worker'];
        legendInfo{3} = ['Average' newline 'percentage'];
    elseif j==3
        legendInfo{1} = ['Fully' newline 'vaccinated '];
        legendInfo{2} = ['Not fully' newline 'vaccinated'];
        legendInfo{3} = ['Average' newline 'percentage'];
    else
        legendInfo{1} = ['Obesity'];
        legendInfo{2} = ['No obesity'];
        legendInfo{3} = ['Average' newline 'percentage'];
    end 
%     legendInfo{1} = ['Unprioritized group'];
%     legendInfo{2} = ['Prioritized group'];
    for i = 1:length(unique_degree)
        y = [y;result{i,j,1},result(i,j,2),result(i,j,3)];
    end 
    %y = y';
    y = cell2mat(y);
    %y = y./sum(y,1);
    y = [result{2,j,1},result{1,j,1}; result{2,j,2},result{1,j,2}; result{2,j,3},result{1,j,3}];
    y = y';
    y = [y;sum(y(1:2,:))];
    %y(2,:) = sum(y);
    y_percentage  = y./sum(y,2);
    h = bar(y_percentage*100 ,'stacked');
    if j == 1
        h(1).FaceColor = [.33 .33 .33];
        h(2).FaceColor = [.66 .66 .66];
        h(3).FaceColor = 'k'; 
    else
        h(1).FaceColor = [.33 .33 .33];
        h(2).FaceColor = 'k';
    end 
    set(gca,'xticklabel',{'Prioritized','Unprioritized','Population'},'Fontsize',13);
    ytickformat('percentage');
    ylabel('Percentage','Fontsize',13);
    total_percentage = sum(y,1)./sum(sum(y,1));
    for i = 1:length(total_percentage)-1
        yline(sum(total_percentage(1:i)*100),'r--','DisplayName','2017');%'Cumulative percentage'
    end 
%     if j == 1
%         set(gca,'xticklabel',{'[0,49]','[50,64]','>65'});
%     elseif j==2
%         set(gca,'xticklabel',{'Healthworker','Not healthworker'});
%     elseif j==3
%         set(gca,'xticklabel',{'Fully vaccinated','Not fully vaccinated'});
%     else
%         set(gca,'xticklabel',{'Obesity','No obesity'});
%     end 
    legend(legendInfo,'Location','northeast','FontSize',13);
    legend boxoff;
    box off;
    cdf_total = 0;
    group_1 = 0;
    group_2 = 0;
    for i = 1:length(total_percentage)
        cdf_total(i) = sum(total_percentage(1:i));
        group_1(i) = sum(y_percentage(1,1:i));
        group_2(i) = sum(y_percentage(2,1:i));
        i
    end 
    t_1(j) = max(abs(cdf_total - group_1));
end 



h = bar(t_1);
h(1).FaceColor = [.33 .33 .33];
set(gca,'xticklabel',{'Age','Occupation','Vaccination','Obesity'},'Fontsize',12);
box off;
low_bound = 1.073;
high_bound = 1.949;
%D_range = [low_bound*sqrt((185+100)/(185*1000)),high_bound*sqrt((185+100)/(185*1000))];
%r = rectangle('Position',[-3 D_range(1) 10 D_range(2)]);
%r.EdgeColor = 'r';
%r.LineWidth = 2;
ylabel('KS statistics');
hline1 = yline(low_bound*sqrt((185+100)/(185*1000)),'--r');
hline2 = yline(high_bound*sqrt((185+100)/(185*1000)),'-r');
legend([hline1,hline2],{'Threshold of \alpha = 0.2','Threshold of \alpha = 0.001'},'Fontsize',12);
legend boxoff;
xlim([0,5]);

%t_2 = max(abs(cdf_total - group_2));


% for i = 1:length(unique_degree)
%     for j = 1:4
%         figure
%         histogram(clu_degree{i}(:,j+1));
%         title(['Histogram of group',num2str(i),' with attributes ', attr(j)]);
%     end
% end   
% 
% for j = 1:4
%     figure
%     histogram(data(:,j+1));
%     title(['Histogram of total with attributes ' +num2str(j)]);
% end 


