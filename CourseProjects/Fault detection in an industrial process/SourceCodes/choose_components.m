%%Stu No:0592964 Name:Subhashree Rautray
%%function calculating best number of components for pca data analysis
function choosed_components_num = choose_components(data, contribution_rate_limit)
    [t,p,r2] =  pca(cov(data)); %r2 is latent, svd decompostion
                                                    %for cov(data)
    
    contribution_percent = 0;           %counter variable used for collecting
                                        %principal component' contributions
    %use the accumulated variance contribution to choose the number of
    %principal components
    D = diag(r2);
    choosed_components_num = 1;
    while sum(D(1:choosed_components_num))/sum(D)<contribution_rate_limit
        choosed_components_num = choosed_components_num + 1;
    end
end
