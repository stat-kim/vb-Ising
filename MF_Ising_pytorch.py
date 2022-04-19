# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:39:33 2022

@author: kmw
"""

import torch
import torch.distributions
import math
import numpy as np
import networkx as nx
import itertools
from scipy.stats import norm
from matplotlib import pyplot as plt
import time 
from sklearn import linear_model
import pandas as pd
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

## Data loading
def read_data(fname):
    data=open(fname,'r')
    M=list()
    for line in data:
       m0=str.split(line,',')
       m0[-1]=str.split(m0[-1],'\n')[0]
       M.append([float(item) for item in m0])
    M=np.array(M)
    return M

class VB_autograd:
    def __init__(self, A,x):
        self.A= A
        self.n= A.shape[0]
        self.x= x
        self.ns= 200
        
    def initial_values(self):
        # Initial values of variational parameters        
        self.mu1= torch.tensor(-0.7*np.ones(self.ns), requires_grad=True) # .to(device)       
        self.mu2= torch.zeros(self.ns, requires_grad=True) # .to(device)       
        self.s1_p= torch.tensor(0.0*np.ones(self.ns), requires_grad=True) # .to(device)       
        self.s2_p= torch.tensor(0.0*np.ones(self.ns), requires_grad=True) # .to(device)       
        
    ## Variational parameters
    def _parameters(self):
        return [self.mu1, self.mu2, self.s1_p, self.s2_p]
    
    ## Hyper parameters of priors
    def _hyper_parameters(self):
        self.mus0= torch.zeros(2)
        self.sigs0= torch.tensor([[1.0,0.0], [0.0,1.0]])
        
    def _log_variational(self, q_samp):
        s1= torch.logaddexp(torch.zeros(self.ns), self.s1_p)
        s2= torch.logaddexp(torch.zeros(self.ns), self.s2_p)
        
        u1= -0.5*torch.log(2*np.pi*(s1**2))
        u2= -((q_samp[:,0] - self.mu1)**2) / (2*(s1**2))
        
        u3= -0.5*torch.log(2*np.pi*(s2**2))
        u4= -((q_samp[:,1] - self.mu2)**2) / (2*(s2**2))
        
        self.log_q= u1 + u2 + u3 + u4

    def _log_prior(self, q_samp):
        self._hyper_parameters()
        
        cov_inv= torch.inverse(self.sigs0)
        cent= self.mus0 - q_samp
        cent_t= torch.transpose(cent,0,1)
        
        u1= torch.matmul(cent.float(),torch.matmul(cov_inv.float(), cent_t.float()))
        u2= torch.diagonal(u1)
        
        v1= -0.5*u2
        v2= 2*torch.tensor(np.pi)*self.sigs0
        v3= -0.5*torch.logdet(v2)

        self.log_p= v1+v3

    def _log_lik(self, q_samp): 
        mx= torch.matmul(self.A, self.x)
        u1= torch.outer(torch.exp(q_samp[:,0]), mx) + q_samp[:,1][:,None]
        u2= u1*x[None,:]
        # u3= u2 - torch.log(torch.cosh(u1))
        
        v1 = torch.sign(u1)*u1
        v2 = torch.exp(-2*v1)
        
        safe_logcosh= v1 + torch.log1p(v2) - torch.log(torch.tensor(2))
        
        u3= u2 - safe_logcosh
        
        
        self.log_likel= torch.sum(u3, 1)-self.n*torch.log(torch.tensor(2)) 
        
    ## ELBO (Objective function)
    def _obj_function(self, q_samp):
        
        self._log_variational(q_samp)
        self._log_prior(q_samp)
        self._log_lik(q_samp)
        
        self.elbo= (self.log_p + self.log_likel) - self.log_q


## Sampling from the current q
def q_sampling(ns, mu1, mu2, s1, s2):
    q1= torch.normal(mean=mu1, std=s1)
    q2= torch.normal(mean=mu2, std=s2)
    temp= torch.zeros(ns,2)
    temp[:,0]= q1
    temp[:,1]= q2
    return temp

A= read_data('A_R_d50_n100.txt') 
A= torch.tensor(A)

x_50= read_data('x_b7_Bneg5_n100_d50.txt')


lr=0.00002
# lr=0.0001
n_epochs= 5000 # number of iterations for updating variational parameters

# Estimates
R= 50
est= torch.zeros(50,2)
results= []
for k in range(R):
    np.random.seed(k)
    torch.manual_seed(k)
    x= torch.tensor(x_50[k,])
    
    fit= VB_autograd(A,x)
    # self=fit
    fit.initial_values()
       
    s1= torch.log(1+torch.exp(fit.s1_p))
    s2= torch.log(1+torch.exp(fit.s2_p))
    
    elbo_hist= torch.zeros(n_epochs)
    mu1_hist= torch.zeros(n_epochs)
    mu2_hist= torch.zeros(n_epochs)
    s1_hist= torch.zeros(n_epochs)
    s2_hist= torch.zeros(n_epochs)
    
    before = time.time()
    for epoch in range(n_epochs):
        
        mus= torch.tensor([fit.mu1[0], fit.mu2[0]])
        # L= torch.tensor([[l1[0],0.0], [fit.l12[0],l2[0]]])
        
        q_samp= q_sampling(fit.ns, fit.mu1[0].detach(), fit.mu2[0].detach(), s1.detach(), s2.detach())
        
        fit._log_variational(q_samp)
        # fit.log_q
        
        fit._log_prior(q_samp)
        # fit.log_p
        
        fit._log_lik(q_samp)
        # fit.log_likel
        
        fit._obj_function(q_samp)
        # fit.elbo
        
        # Parameter updates
        
        # Update of mu1
        external_grad= torch.ones(fit.log_q.shape)
        fit.log_q.backward(gradient=external_grad)
        
        fit.mu1= fit.mu1.detach() + lr*torch.mean(fit.mu1.grad*fit.elbo.detach())
        fit.mu1.requires_grad=True
        
        # Update of mu2
        fit.mu2= fit.mu2.detach() + lr*torch.mean(fit.mu2.grad*fit.elbo.detach())
        fit.mu2.requires_grad=True
        
        # Update of s1_p
        fit.s1_p= fit.s1_p.detach() + lr*torch.mean(fit.s1_p.grad*fit.elbo.detach())
        fit.s1_p.requires_grad=True
        
        # Update of s2_p
        fit.s2_p= fit.s2_p.detach() + lr*torch.mean(fit.s2_p.grad*fit.elbo.detach())
        fit.s2_p.requires_grad=True
        
        s1= torch.logaddexp(torch.zeros(fit.ns), fit.s1_p)
        s2= torch.logaddexp(torch.zeros(fit.ns), fit.s2_p)
        
        elbo_hist[epoch]= torch.mean(fit.elbo).item()
        mu1_hist[epoch]= fit.mu1[0].item()
        mu2_hist[epoch]= fit.mu2[0].item()
        s1_hist[epoch]= s1[0].item()
        s2_hist[epoch]= s2[0].item()
    
    plt.plot(elbo_hist)
    # plt.plot(l2_hist)
    # plt.plot(np.exp(mu1_hist))
    after = time.time()
    print(after - before)
    print(k)
    
    est[k,0]= torch.exp(mu1_hist[-1])
    est[k,1]= mu2_hist[-1]
    
    results.append([elbo_hist, mu1_hist, mu2_hist, s1_hist, s2_hist])
    

beta0= 0.7
B0= -0.5
print(torch.mean((est[:,0] - beta0)**2 + (est[:,1] - B0)**2))

# len(results)
# len(results[0])
# plt.plot(torch.exp(results[0][2]))
# torch.save(results, 'Results_MF_be7_Bneg5_n100_d50_S200.pt')










#########
#########
###
###
###
class VB_autograd:
    def __init__(self, A,x):
        self.A= A
        self.n= A.shape[0]
        self.x= x
        self.ns= 200
        
    def initial_values(self):
        # Initial values of variational parameters        
        self.mu1= torch.tensor(-0.7*np.ones(self.ns), requires_grad=True) # .to(device)       
        self.mu2= torch.zeros(self.ns, requires_grad=True) # .to(device)       
        self.s1_p= torch.tensor(-0.2*np.ones(self.ns), requires_grad=True) # .to(device)       
        self.s2_p= torch.tensor(-0.2*np.ones(self.ns), requires_grad=True) # .to(device)       
        
    ## Variational parameters
    def _parameters(self):
        return [self.mu1, self.mu2, self.s1_p, self.s2_p]
    
    ## Hyper parameters of priors
    def _hyper_parameters(self):
        self.mus0= torch.zeros(2)
        self.sigs0= torch.tensor([[1.0,0.0], [0.0,1.0]])
        
    def _log_variational(self, q_samp):
        s1= torch.logaddexp(torch.zeros(self.ns), self.s1_p)
        s2= torch.logaddexp(torch.zeros(self.ns), self.s2_p)
        
        u1= -0.5*torch.log(2*np.pi*(s1**2))
        u2= -((q_samp[:,0] - self.mu1)**2) / (2*(s1**2))
        
        u3= -0.5*torch.log(2*np.pi*(s2**2))
        u4= -((q_samp[:,1] - self.mu2)**2) / (2*(s2**2))
        
        self.log_q= u1 + u2 + u3 + u4

    def _log_prior(self, q_samp):
        self._hyper_parameters()
        
        cov_inv= torch.inverse(self.sigs0)
        cent= self.mus0 - q_samp
        cent_t= torch.transpose(cent,0,1)
        
        u1= torch.matmul(cent.float(),torch.matmul(cov_inv.float(), cent_t.float()))
        u2= torch.diagonal(u1)
        
        v1= -0.5*u2
        v2= 2*torch.tensor(np.pi)*self.sigs0
        v3= -0.5*torch.logdet(v2)

        self.log_p= v1+v3

    def _log_lik(self, q_samp): 
        mx= torch.matmul(self.A, self.x)
        u1= torch.outer(torch.exp(q_samp[:,0]), mx) + q_samp[:,1][:,None]
        u2= u1*x[None,:]
        # u3= u2 - torch.log(torch.cosh(u1))
        
        v1 = torch.sign(u1)*u1
        v2 = torch.exp(-2*v1)
        
        safe_logcosh= v1 + torch.log1p(v2) - torch.log(torch.tensor(2))
        
        u3= u2 - safe_logcosh
        
        
        self.log_likel= torch.sum(u3, 1)-self.n*torch.log(torch.tensor(2)) 
        
    ## ELBO (Objective function)
    def _obj_function(self, q_samp):
        
        self._log_variational(q_samp)
        self._log_prior(q_samp)
        self._log_lik(q_samp)
        
        self.elbo= (self.log_p + self.log_likel) - self.log_q


A= torch.load('A_facebook.pt') 

# x= torch.load('x_birthday.pt')
x= torch.load('x_gender.pt')
# x= torch.load('x_location.pt')
# x= torch.load('x_school.pt')
sum(x== 1) # model size

n_epochs= 10000 # number of iterations for updating variational parameters
lr=0.00001

fit= VB_autograd(A,x)
# self=fit
fit.initial_values()
   
s1= torch.log(1+torch.exp(fit.s1_p))
s2= torch.log(1+torch.exp(fit.s2_p))

elbo_hist= torch.zeros(n_epochs)
mu1_hist= torch.zeros(n_epochs)
mu2_hist= torch.zeros(n_epochs)
s1_hist= torch.zeros(n_epochs)
s2_hist= torch.zeros(n_epochs)

np.random.seed(1)
torch.manual_seed(1)


before = time.time()
for epoch in range(n_epochs):
        
        mus= torch.tensor([fit.mu1[0], fit.mu2[0]])
        # L= torch.tensor([[l1[0],0.0], [fit.l12[0],l2[0]]])
        
        q_samp= q_sampling(fit.ns, fit.mu1[0].detach(), fit.mu2[0].detach(), s1.detach(), s2.detach())
        
        fit._log_variational(q_samp)
        # fit.log_q
        
        fit._log_prior(q_samp)
        # fit.log_p
        
        fit._log_lik(q_samp)
        # fit.log_likel
        
        fit._obj_function(q_samp)
        # fit.elbo
        
        # Parameter updates
        
        # Update of mu1
        external_grad= torch.ones(fit.log_q.shape)
        fit.log_q.backward(gradient=external_grad)
        
        fit.mu1= fit.mu1.detach() + lr*torch.mean(fit.mu1.grad*fit.elbo.detach())
        fit.mu1.requires_grad=True
        
        # Update of mu2
        fit.mu2= fit.mu2.detach() + lr*torch.mean(fit.mu2.grad*fit.elbo.detach())
        fit.mu2.requires_grad=True
        
        # Update of s1_p
        fit.s1_p= fit.s1_p.detach() + lr*torch.mean(fit.s1_p.grad*fit.elbo.detach())
        fit.s1_p.requires_grad=True
        
        # Update of s2_p
        fit.s2_p= fit.s2_p.detach() + lr*torch.mean(fit.s2_p.grad*fit.elbo.detach())
        fit.s2_p.requires_grad=True
        
        s1= torch.logaddexp(torch.zeros(fit.ns), fit.s1_p)
        s2= torch.logaddexp(torch.zeros(fit.ns), fit.s2_p)
        
        elbo_hist[epoch]= torch.mean(fit.elbo).item()
        mu1_hist[epoch]= fit.mu1[0].item()
        mu2_hist[epoch]= fit.mu2[0].item()
        s1_hist[epoch]= s1[0].item()
        s2_hist[epoch]= s2[0].item()
        
        if (epoch+1)%500 == 0:
            print(epoch+1)

after = time.time()
print(after - before)

plt.plot(elbo_hist)
plt.plot(mu1_hist)
plt.plot(mu2_hist)
plt.plot(s1_hist)
plt.plot(s2_hist)
# s1_hist[-1]
# s2_hist[-1]

opt_idx= torch.argmax(elbo_hist)
print(torch.exp(mu1_hist[opt_idx]))
print(mu2_hist[opt_idx])
s1_hist[opt_idx]
s2_hist[opt_idx]

result= [elbo_hist, mu1_hist, mu2_hist, s1_hist, s2_hist]
torch.save(result, 'Results_MF_facebook_gender.pt')

##
mu_opt= mu2_hist[opt_idx]
s_opt= s2_hist[opt_idx]

np.random.seed(123)
torch.manual_seed(123)
draws= torch.normal(mean=mu_opt, std=s_opt, size=(264,)) # gender: 264, school: 747

# print(torch.std(torch.exp(draws)))
print(torch.std(draws))



## histogram
result= torch.load('Results_MF_facebook_gender.pt')

opt_idx= torch.argmax(result[0])
mu1_opt= result[1][opt_idx]
mu2_opt= result[2][opt_idx]
s1_opt= result[3][opt_idx]
s2_opt= result[4][opt_idx]

np.random.seed(123)
torch.manual_seed(123)

d= torch.normal(mean=mu1_opt, std=s1_opt, size=(10000,)) # gender: 264, school: 747
d= torch.exp(d)

hist, bin_edges= np.histogram(d)

n, bins, patches = plt.hist(x=d, bins=20, color='green',
                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Draws from MF family (School)')
plt.text(0.45, 80, r'$\beta$', fontsize=25)
# plt.text(-1.33, 70, r'$B$', fontsize=25)



