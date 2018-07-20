# neighborhood-selection-with-lrLMM

Report on an Application of the Neighborhood Selection

1. Overview
The neighborhood selection is known to be a computationally attractive alternative to perform covariance selection, which estimate the conditional independence restrictions of any given node in the graph and is hence equivalent to variable selection for Gaussian linear models. In this report, we apply the neighborhood selection with improved methods of Lasso to analyze an economic performance, trying to find some associations between financial variables related to a bank’s performance.

2. Introduction of the Application
We are intended to apply the neighborhood selection to analyze an economic performance, trying to find associations between financial variables. To examine the correctness and accuracy of our attempt, we also conduct a simulation by generating data from a multivariate normal distribution. The real economic data is collected from 13 branches of Fentai subbranch with the performances of 19 variables from a period of 3 months separately.

At first, we applied classical Lasso to perform the neighborhood selection, which presented an unsatisfactory result because the data is non-i.i.d. Since the source of the data came from different regions of different periods of time, dependencies can be introduced. We then improve the method by introducing Truncated-rank Sparse Linear Mixed Model (TrSLMM), which is a unified framework for sparse variable selection in heterogeneous datasets.

3. Related Work
Least Absolute Shrinkage and Selection Operator (Lasso) is a regression analysis that performs variable selection and regularization by altering the model fitting process to select only a subset of the provided covariates for use in the final model rather than using all of them [1]. By adding a penalty to force the sum of the absolute value of the regression coefficients to be less than a fixed value, Lasso constrains some coefficients to zero so that the dimension of the data can be reduced and at the same time, the regression coefficients will not be too large. Given a sample consisted of N cases. Let yi be the single outcome and X be the covariance matrix. The aim of Lasso is to solve
 
where   is the standard   norm. 

In classical linear mixed models, the covariance between observations is denoted by K, i.e., when K = 0, variables are independent from one another and classical Lasso can work just fine. However, when the data is collected from different sources, such as different regions, populations, time, etc., K ≠ 0 and dependencies can be introduced between variables in such a heterogeneous dataset, from which classical Lasso may fail to find the correct association between variables.

To copy with heterogeneous datasets, we apply Truncated-rank Sparse Linear Mixed Model (TrSLMM) to improve our performance [2]. Instead of simply applying K = XXT as the estimate the covariance between observations, which might be full-rank, TrSLMM seeks a low-rank approximation of K. Let Γ := XXT and Γ = UΛVT be the SVD of Γ. By directly screening out the top, dominant singular values, relatively important information of the matrix can be preserved. By selecting the top s values Λj for which 
 
where n is the number of samples, then we have:
 
and:
 
After some confounder correction, classical methods of variable selection can be applied to modified, rotated data:
 

The neighborhood selection is a computationally attractive alternative to perform covariance selection, i.e., to find the structural zeros from data [3]. It estimates the conditional independence restrictions of any given node in the graph and is hence equivalent to variable selection for Gaussian linear models. A graphical model is constructed to represent the associations between each variable in the neighborhood selection, as each node in the graph represents a variable and node a shares an edge with node b only when node a is conditionally dependent on node b, given all remaining variables. As a generalization of the condition given all remaining variables, an optimal prediction, given only the subset of variables, is aimed to solve
 
where the set of nonzero coefficients is identical to the set of nonzero entries in the corresponding row vector of the inverse covariance matrix and defines precisely the set of neighbors of node A. Therefore, the set of neighborhoods of node a can be written as:
 
where Γ(n) is the node set in the graph. In other words, when predicting a variable Xa given all remaining variables, the zero coefficients correspond to each pair of variables not contained in the edge set of the graph. 

4. Simulation
To examine the correctness and accuracy of our attempt to find associations between variables with the neighborhood selection, we conduct a simulation by generating data from a multivariate normal distribution. We first determine a positive-define diagonal matrix as the inverse covariance matrix, i.e., the precision matrix, to generate a multivariate normal distribution, which is our simulation data. Then we apply neighborhood selection with TrSLMM to conduct the synthetic experiment. 

Since the set of neighborhood of node A is identical to the nonzero entries in the corresponding row vector of the inverse covariance matrix of a multivariate normal distribution, which corresponds to conditional independence restrictions between variables, we compare the intersection between the result of neighborhood selection and the original precision matrix to examine the correctness and accuracy of our attempt.

5. Experiment
We apply the neighborhood selection to analyze an economic performance, trying to find associations between financial variables collected from a bank called Fengtai subbranch. The data is consisted of 13 branches of Fentai subbranch with the performances of 19 variables collected from December, January to February, each of which contains 399, 393 and 217 samples separately, 1009 in all. The 19 variables include large deposits, insurance, funding, installment, product, payment and loan via mobile phone. We make use of the Matlibplot, a plotting library in Python to visualize the result in the form of two different layouts – circular and spring layouts. 

The results of the neighborhood selection with classical Lasso are as follows:
                                
                                  
The result we obtained is not satisfactory because the graphs are not sparse enough, i.e., the neighborhood selection with classical Lasso fails to estimate the associations between variables properly, since our dataset was collected from different regions and dates, which is a heterogeneous dataset. Therefore, we then applied TrSLMM to improve the performance and made use of the rotated data generated from the TrSLMM and applied it to adaptive Lasso. The results are as follows:
                               
                            

6. Conclusion
We are intended to analyze and estimate the associations between 19 economic variables in the performance of a bank. We first applied the neighborhood selection with classical Lasso, yet the classical Lasso fails to learn the relationships between variables since the data was collected from different regions of different periods of time and dependencies are introduced between observations. To copy with heterogeneous dataset like this, we introduced TrSLMM with cross validation to improve the performance. We also made use of the rotated data generated from TrSLMM so that the modified data can be better applied to the classical Lasso.

[1] Tibshirani, Robert (1996). “Regression Shrinkage and Selection via the lasso”. Journal of the Royal Statistical Society. Series B (methodological). Wiley. 58 (1): 267–88. JSTOR 2346178.
[2] Haohan Wang, Bryon Aragam and Eric P. Xing (2017). “Variable Selection in Heterogeneous Datasets: A Truncated-rank Sparse Linear Mixed Model with Applications to Genome-wide Association Studies”. bioRxiv preprint first posted online Dec. 3, 2017; doi: http://dx.doi.org/10.1101/228106.
[3] Nicolai Meinshausen, Peter B¨uhlmann (2006). “High-dimensional Graphs and Variable Selection with the Lasso”. The Annals of Statistics 2006, Vol. 34, No. 3, 1436–1462 DOI: 10.1214/009053606000000281,Institute of Mathematical Statistics

