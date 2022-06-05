%Stu No:589715 Name:Yang Tuo Stu No:0592964 Name:Subhashree Rautray
%perpare for using dataset d04_te,d09_te and d15_te for data analysis
%use d01_te as the reference
clc,clear,close all;
%use d00_te.dat as the reference to devitate fault behaviours from other
%datasets
load ./ADAML-data/d00_te.dat  %observations under normal conditions
load ./ADAML-data/d04_te.dat
load ./ADAML-data/d11_te.dat
load ./ADAML-data/d15_te.dat

fault_dataset = {'Fault4',d04_te;...
    'Fault11',d11_te;...
    'Fault15',d15_te};
contribution_rate_limit = 0.85;  %use this limit to confirm the number of 
                          %principle components we will choose
pca_error_positions = [161,165;356,167;100,293];

%use PCA and DPCA methods to resepctively analyse three datasets
for k = 1:length(fault_dataset)
    fault_data = fault_dataset{k,2};
    scaled_fault_data = scale(fault_data);   %standardise the data
    normal_data = d00_te;
    scaled_normal_data = scale(d00_te);

    %choose the number of components(normal data can be seen as train data,
    %fault_data can be seen as test data,use train data to get the number of
    %principal components)
    pca_choosed_components_num = choose_components(scaled_normal_data,...
            contribution_rate_limit);
%     pca_choosed_components_num = 12;

    %calculate the T2 values for the PCA method 
    [T2,T2_threshold] = T2_calculation(scaled_normal_data,scaled_fault_data,...
        size(fault_data,1),pca_choosed_components_num,0.05); %

    gca = figure;
    plot(T2,'b-');
    hold on;
    plot(repmat(T2_threshold,1,size(scaled_fault_data,1)),'r--');
    title(fault_dataset{k,1});
    xlabel('time(hr)');
    ylabel('T^2 PCA');
    legend('Fault trend',['T^2 threshold:',num2str(T2_threshold)],'Location',...
        'Best');
    saveas(gca,['./figures/',fault_dataset{k,1},'_T2_PCA.fig']);

    %calculate the Q2 values for the PCA method
    data = (fault_data-mean(normal_data))./std(normal_data);
    [Q,Q_threshold] = Q_calculation(scaled_normal_data,data,size(fault_data,1),...
        pca_choosed_components_num,0.008);

    gca = figure;
    plot(Q,'b-');
    hold on;
    plot(repmat(Q_threshold,1,size(scaled_fault_data,1)),'r--');
    title(fault_dataset{k,1});
    xlabel('time(hr)');
    ylabel('Q PCA');
    legend('Fault trend',['Q threshold:',num2str(Q_threshold)],'Location',...
        'Best');
    saveas(gca,['./figures/',fault_dataset{k,1},'_Q_PCA.fig']);

    %calculate the accumulated contribution rates
    %and draw contribution rate
    %for T2 statistics
    %1. confirm the score resulting in out of control
    [X_row,X_col] = size(scaled_normal_data);
    sigmaXtrain = cov(scaled_normal_data);
    [T,lamda] = eig(sigmaXtrain);
    D = diag(lamda);
    P = T(:,X_col-pca_choosed_components_num+1:X_col);
    [r,y] = size(P*P');
    I = eye(r,y);
    S = scaled_fault_data(pca_error_positions(k,1),:)*...
        P(:,1:pca_choosed_components_num);
    r = [];
    for m = 1:pca_choosed_components_num
        if S(m)^2/lamda(m) > T2/pca_choosed_components_num
            r = cat(2,r,m);
        end
    end
    %2 calculate each variable's contribution for the last
    %score
    cont = zeros(length(r),X_col);
    for i = length(r)
        for j = 1:X_col
            cont(i,j) = abs(S(i)/D(i)*P(j,i)*...
                scaled_fault_data(pca_error_positions(k,1),j));
        end
    end

    %calculate overall contributions of all variables
    CONTJ = zeros(X_col,1);
    for j = 1:X_col
       CONTJ(j) = sum(cont(:,j));
    end

    %4.calculate each variable' contributions for Q
    e = scaled_fault_data(pca_error_positions(k,1),:)*(I-P*P');
    contq = e.^2;

    %5.draw the contribution map
    gca = figure;
    subplot(2,1,1);
    bar(CONTJ,'k');
    title('Variable contributions for T2 values(PCA method)');
    xlabel('variable No');
    ylabel('T^2 contributions');

    subplot(2,1,2);
    bar(contq,'k');
    title('Variable contributions for Q values(PCA method)');
    xlabel('variable No');
    ylabel('Q contributions');
    saveas(gca,['./figures/',fault_dataset{k,1},'_contributions.fig']);
    
    %calculate the mixed statistics combined with T2 and Q for PCA methods
    %1.caculate the threshold value
    alpha = 0.9;
    S = lamda(X_col-pca_choosed_components_num+1:X_col,...
        X_col-pca_choosed_components_num+1:X_col);
    FAI = P*pinv(S)*P'/T2_threshold+(eye(X_col)-P*P')/Q_threshold;
    S = cov(scaled_normal_data);
    g = trace((S*FAI)^2)/trace(S*FAI);
    h = (trace(S*FAI))^2/trace((S*FAI)^2);
    kesi = g*chi2inv(alpha,h);
    %%comprehensive(combined) statistics
    gca = figure;
    fai = (Q/Q_threshold)+(T2/T2_threshold);
    plot(fai,'b-');
    hold on;
    plot(repmat(kesi,1,size(scaled_fault_data,1)),'r--');
    title(['mixed statistics:(',num2str(fault_dataset{k,1}),') for PCA method']);
    xlabel('time(hr)');
    ylabel('mixed statistics combining T2 and Q');
    legend('fault trend',['threshold value:',num2str(kesi)]);
    saveas(gca,['./figures/',fault_dataset{k,1},'_mixed_statistics_PCA.fig']);
    
    %%DPCA method
    history_num = 4;    %use the fronter four moments data to spin at the row end
                        %of the matrix
    [row,column] = size(normal_data);   %get the row and column info of the matrix
    normal_data_dpca_matrix = zeros(row-history_num, (history_num+1)*column);
    fault_data_dpca_matrix = zeros(row-history_num, (history_num+1)*column);

    %use the before four moments' data as history data,and put the data of
    %current moment at the start of each row, so the number of total columns
    %DPCA matrix will be (history_num+1)*column),and the number of total rows
    %will be row-history_num
    for i = history_num+1:row   %i represents the row
        for j = 0:history_num
            normal_data_dpca_matrix(i-history_num,j*column+1:(j+1)*column) = normal_data(i-j,:);
            fault_data_dpca_matrix(i-history_num,j*column+1:(j+1)*column) = fault_data(i-j,:);
        end
    end

    %auto scaling
    dpca_normal_data = scale(normal_data_dpca_matrix);
    dpca_fault_data =  scale(fault_data_dpca_matrix);

    %choose the number of components
    %   dpca_choosed_components_num = choose_components(dpca_normal_data, contribution_limit);
    dpca_choosed_components_num = 26;

    %T2 calculation for DPCA
    %calculate the T2 values for the DPCA method 
    [T2,T2_threshold] = T2_calculation(dpca_normal_data,dpca_fault_data,row,...
    dpca_choosed_components_num,0.05);

    gca = figure;
    plot(T2,'b-');
    hold on;
    plot(repmat(T2_threshold,1,size(scaled_fault_data,1)),'r--');
    title(fault_dataset{k,1});
    xlabel('time(hr)');
    ylabel('T^2 DPCA');
    legend('Fault trend',['T2 threshold:',num2str(T2_threshold)],'Location',...
        'Best');
    saveas(gca,['./figures/',fault_dataset{k,1},'_T2_DPCA.fig']);

    %calculate the Q2 values for the DPCA method
    data = (fault_data_dpca_matrix-mean(normal_data_dpca_matrix))./...
        std(normal_data_dpca_matrix);
    [Q,Q_threshold] = Q_calculation(dpca_normal_data,data,row,...
        dpca_choosed_components_num,0.01);

    gca = figure;
    plot(Q,'b-');
    hold on;
    plot(repmat(Q_threshold,1,size(scaled_fault_data,1)),'r--');
    title(fault_dataset{k,1});
    xlabel('time(hr)');
    ylabel('Q DPCA');
    legend('Fault trend',['Q threshold:',num2str(Q_threshold)],'Location',...
        'Best');
    saveas(gca,['./figures/',fault_dataset{k,1},'_Q_DPCA.fig']);
    
    %calculate the mixed statistics combined with T2 and Q for DPCA methods
    %1.caculate the threshold value
    [X_row,X_col] = size(dpca_normal_data);
    sigmaXtrain = cov(dpca_normal_data);
    [T,lamda] = eig(sigmaXtrain);
    D = diag(lamda);
    P = T(:,X_col-dpca_choosed_components_num+1:X_col);
    alpha = 0.9;
    S = lamda(X_col-dpca_choosed_components_num+1:X_col,...
        X_col-dpca_choosed_components_num+1:X_col);
    FAI = P*pinv(S)*P'/T2_threshold+(eye(X_col)-P*P')/Q_threshold;
    S = cov(dpca_normal_data);
    g = trace((S*FAI)^2)/trace(S*FAI);
    h = (trace(S*FAI))^2/trace((S*FAI)^2);
    kesi = g*chi2inv(alpha,h);
    %%comprehensive(combined) statistics
    gca = figure;
    fai = (Q/Q_threshold)+(T2/T2_threshold);
    plot(fai,'b-');
    hold on;
    plot(repmat(kesi,1,size(scaled_fault_data,1)),'r--');
    title(['Mixed statistics:(',num2str(fault_dataset{k,1}),')for DPCA method']);
    xlabel('time(hr)');
    ylabel('mixed statistics combining T2 and Q');
    legend('fault trend',['threshold value:',num2str(kesi)]);
    saveas(gca,['./figures/',fault_dataset{k,1},'_mixed_statistics_DPCA.fig']);
end
%the best 








    
    






