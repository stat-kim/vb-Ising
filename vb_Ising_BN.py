import torch
import torch.distributions
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import time 

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
        self.mu1= torch.tensor(-0.7*np.ones(self.ns), requires_grad=True)  
        self.mu2= torch.zeros(self.ns, requires_grad=True) 
        self.l1_p= torch.tensor(0.0*np.ones(self.ns), requires_grad=True) 
        self.l2_p= torch.tensor(0.0*np.ones(self.ns), requires_grad=True) 
        self.l12= torch.tensor(0.01*np.ones(self.ns), requires_grad=True) 
        
    ## Variational parameters
    def _parameters(self):
        return [self.mu1, self.mu2, self.l1_p, self.l2_p, self.l12]
    
    ## Hyper parameters of priors
    def _hyper_parameters(self):
        self.mus0= torch.zeros(2)
        self.sigs0= torch.tensor([[1.0,0.0], [0.0,1.0]])
        
    def _log_variational(self, q_samp):
        # l1= torch.log(1+torch.exp(self.l1_p))
        # l2= torch.log(1+torch.exp(self.l2_p))
        
        l1= torch.logaddexp(torch.zeros(fit.ns), self.l1_p)
        l2= torch.logaddexp(torch.zeros(fit.ns), self.l2_p)
        
        u1= -torch.log(2*np.pi*l1*l2)
        u2= ((self.l12**2)+(l2**2))*((q_samp[:,0] - self.mu1)**2)
        u3= (2*l1*self.l12)*(q_samp[:,0] - self.mu1)*(q_samp[:,1] - self.mu2)
        u4= (l1**2)*((q_samp[:,1] - self.mu2)**2)
        
        v1= (u2-u3+u4) / ((l1*l2)**2)
        v2= -0.5*v1
        
        self.log_q= u1+v2

    def _log_prior(self, q_samp):
        self._hyper_parameters()
        
        cov_inv= torch.inverse(self.sigs0)
        cent= self.mus0-q_samp
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
def q_sampling(ns, mus, L):
    cov= torch.matmul(L, torch.transpose(L, 0, 1))
    
    return torch.tensor(np.random.multivariate_normal(mus, cov, ns))

A= read_data('A_R_d10_n100.txt')   # Copuling matrix
A= torch.tensor(A)

x_50= read_data('x_b2_B2_n100_d10.txt') # Observed vector x

lr=0.00004
n_epochs= 5000 # number of iterations for updating variational parameters

# Estimates
R= 50
est= torch.zeros(R,2)
results= []
for k in range(R):
    np.random.seed(k)
    torch.manual_seed(k)
    
    x= torch.tensor(x_50[k,])
    
    fit= VB_autograd(A,x)
    # self=fit
    fit.initial_values()
       
    l1= torch.logaddexp(torch.zeros(fit.ns), fit.l1_p)
    l2= torch.logaddexp(torch.zeros(fit.ns), fit.l2_p)
    
    elbo_hist= torch.zeros(n_epochs)
    mu1_hist= torch.zeros(n_epochs)
    mu2_hist= torch.zeros(n_epochs)
    l1_hist= torch.zeros(n_epochs)
    l2_hist= torch.zeros(n_epochs)
    l12_hist= torch.zeros(n_epochs)
    
    before = time.time()
    for epoch in range(n_epochs):
        
        mus= torch.tensor([fit.mu1[0], fit.mu2[0]])
        L= torch.tensor([[l1[0],0.0], [fit.l12[0],l2[0]]])
        
        q_samp= q_sampling(fit.ns, mus.detach(), L.detach())
        
        fit._log_variational(q_samp)
        # fit.log_q
        
        fit._log_prior(q_samp)
        # fit.log_p
        
        fit._log_lik(q_samp)
        # fit.log_likel
        
        fit._obj_function(q_samp)
        # fit.elbo
        
        # Parameter updates
        external_grad= torch.ones(fit.log_q.shape)
        fit.log_q.backward(gradient=external_grad)
        
        # Update of mu1
        fit.mu1= fit.mu1.detach() + lr*torch.mean(fit.mu1.grad*fit.elbo.detach())
        fit.mu1.requires_grad=True
        
        # Update of mu2
        fit.mu2= fit.mu2.detach() + lr*torch.mean(fit.mu2.grad*fit.elbo.detach())
        fit.mu2.requires_grad=True
        
        # Update of l1_p
        fit.l1_p= fit.l1_p.detach() + lr*torch.mean(fit.l1_p.grad*fit.elbo.detach())
        fit.l1_p.requires_grad=True
        
        # Update of l2_p 
        fit.l2_p= fit.l2_p.detach() + lr*torch.mean(fit.l2_p.grad*fit.elbo.detach())
        fit.l2_p.requires_grad=True
        
        # Update of l12  
        fit.l12= fit.l12.detach() + lr*torch.mean(fit.l12.grad*fit.elbo.detach())
        fit.l12.requires_grad=True
        
        l1= torch.logaddexp(torch.zeros(fit.ns), fit.l1_p)
        l2= torch.logaddexp(torch.zeros(fit.ns), fit.l2_p)
        
        elbo_hist[epoch]= torch.mean(fit.elbo).item()
        mu1_hist[epoch]= fit.mu1[0].item()
        mu2_hist[epoch]= fit.mu2[0].item()
        l1_hist[epoch]= l1[0].item()
        l2_hist[epoch]= l2[0].item()
        l12_hist[epoch]= fit.l12[0].item()

    after = time.time()
    print(after - before)
    print(k)
    
    # est[k,0]= torch.exp(mu1_hist[-1])
    # est[k,1]= mu2_hist[-1]
    
    results.append([elbo_hist, mu1_hist, mu2_hist, l1_hist, l2_hist, l12_hist])
