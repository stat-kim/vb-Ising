## Step 1: Download observed data.
In order to implement our variational Bayes (VB) algorithm for Ising model parameter estimation, a coupling matrix $A_n$ and a binary vector $\boldsymbol{x}$ are needed. Pre-generated coupling matrices and binary vectors are stored in the directory "./ObservedData". The files are named as "A_(1)\_d(2)\_n(3).txt" for a copuling matrix and as "x\_(1)\_d(2)\_n(3)\_beta(4)\_B(5).txt" for a binary vector where (1) stands for the type of the underlying graph (regular or irregluar), (2) stands for corresponding degree of the graph $d$, (3) represents size of a binary vector or a coupling matrix, (4) and (5) represent the true values of the parameters $(\beta_0, B_0)$. For examples, the file "A_RG_d20_n100.txt" contains a regular coupling matrix $A_n$ of size 100 by 100 with $d=20$ and the file "x_IRG_d44_n500_beta7_Bneg2.txt" contains 50 binary vectors generated with an irregular coupling matrix of size 500 by 500, $d_n=44$, and $(\beta_0 = 0.7, B_0 = -0.2)$. We also provide the coupling matrix and the binary vector we used for Facebook network data analysis stored in the files "A_facebook.pt" and "x_gender.pt" respectively.

One can use the python file, `Data_generating_coupling_matrix_observation.py`, to generate more data.



## Step 2: Run python files.
You can see two python files for implementation of our VB algoritms. The file `vb_Ising_MF.py` contains codes for VB algorithm with mean-field family and the file `vb_Ising_BN.py` contains codes with bivariate Gaussian family where necessary comments are provided. For other methods, PMLE and MCMC, see the files `Other_methods.py`.
