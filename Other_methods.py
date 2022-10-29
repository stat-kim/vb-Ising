import torch
import torch.distributions
import numpy as np
import time 

## Define functions
def read_data(fname):
    data=open(fname,'r')
    M=list()
    for line in data:
       m0=str.split(line,',')
       m0[-1]=str.split(m0[-1],'\n')[0]
       M.append([float(item) for item in m0])
    M=np.array(M)
    return M

def Ising_pmle(x,A):
    # n=len(x)
    mx= torch.matmul(A,x)
    betastep= torch.linspace(0.01, 2.0,steps=200)
    Bstep= torch.linspace(-1,1,steps=201)
    PS1mat= torch.zeros(len(betastep),len(Bstep))
    PS2mat= torch.zeros(len(betastep),len(Bstep))
    minvec= torch.zeros(2)
    minimum= 50
    PSmat= torch.sqrt(PS1mat**2 + PS2mat**2)
    for i in range(len(betastep)):
        for j in range(len(Bstep)):
            PS1mat[i,j]= torch.sum(mx*x - mx*torch.tanh(betastep[i]*mx + Bstep[j]))
            PS2mat[i,j]= torch.sum(x - torch.tanh(betastep[i]*mx +Bstep[j]))
            PSmat[i,j]= torch.sqrt((PS1mat[i,j])**2 + (PS2mat[i,j])**2)
            if PSmat[i,j] < minimum:
                minvec= torch.tensor((i,j))
                minimum= PSmat[i,j]
    
    return torch.tensor((betastep[int(minvec[0])], Bstep[int(minvec[1])]))

def Ising_sampling(A, theta, burnin=100):
    n= A.shape[0]    
    beta= theta[0]
    B= theta[1]
    z= 2.0*torch.bernoulli(0.5*torch.ones(n))-1
    
    for iter in range(burnin):
        mz= torch.matmul(A.float(), z.float())
        v= beta*mz + B
        
        prob= torch.exp(v - torch.logaddexp(v, -v))
        # prob = torch.exp(v) / (torch.exp(v) + torch.exp(-v))
        z= 2.0*torch.bernoulli(prob) - 1
    return z

def log_Ising_unnormalized(A,x,theta):
    beta= theta[0]
    B= theta[1]
    v1= 0.5*beta*torch.matmul(x.float(), torch.matmul(A.float(), x.float())) + B*torch.sum(x)
    return v1

######################################################################

## MCMC
A= read_data('A_IRG_d20_n100.txt') # Coupling matrix
A= torch.tensor(A)

x_50= read_data('x_IRG_d20_n100_beta2_B2.txt') # Binary vertors

results= []
np.random.seed(1)
torch.manual_seed(1)

R= 50
for k in range(R):
    observed_x= torch.tensor(x_50[k,]) 
    
    before= time.time()
    pmle= Ising_pmle(observed_x, A) # initial guess
    current_theta= pmle.clone()
    mcmc_all_chain= []
    
    nstep= 20000
    for step in range(nstep):
        current_x= Ising_sampling(A, current_theta)
        proposal_theta= torch.normal(mean= current_theta, std= torch.tensor([0.1, 0.1]))
        
        if proposal_theta[0] > 0:
            auxiliary_x= Ising_sampling(A, proposal_theta)
            
            u1= log_Ising_unnormalized(A, auxiliary_x, pmle)
            u2= log_Ising_unnormalized(A, observed_x, proposal_theta)
            u3= log_Ising_unnormalized(A, current_x, current_theta)
            
            v1= log_Ising_unnormalized(A, current_x, pmle)
            v2= log_Ising_unnormalized(A, observed_x, current_theta)
            v3= log_Ising_unnormalized(A, auxiliary_x, proposal_theta)
            
            H= torch.exp((u1+u2+u3) - (v1+v2+v3))
        
            if H.float() > torch.rand(1):
                current_theta= proposal_theta.clone()
        
            mcmc_all_chain.append(proposal_theta)
    
    after= time.time()
    print(after - before)
    results.append([mcmc_all_chain])

# torch.save(results, 'Results_MCMC_be2_B2_n100_d20_irregular.pt') # Save the results

# Calculate MSE
est_R= torch.zeros(len(results),2)
for r in range(len(results)):
    est= torch.zeros(len(results[r][0]),2)
    for i in range(len(results[r][0])):
        est[i,:]= results[r][0][i]
    
    est_R[r,:]= torch.mean(est[10000:,:],0)

beta0= 0.2
B0= 0.2

print(torch.mean((est_R[:,0] - beta0)**2 + (est_R[:,1] - B0)**2))

######################################################################

## PMLE
A= read_data('A_IRG_d20_n100.txt') # Coupling matrix
A= torch.tensor(A)

x_50= read_data('x_IRG_d20_n100_beta2_B2.txt') # Binary vertors

R=50
est= torch.zeros(R,2)
for k in range(R):
    before= time.time()
    x= torch.tensor(x_50[k,])
    est[k,]= Ising_pmle(x,A)
    after = time.time()
    print(after - before)

beta0= 0.2
B0= 0.2
print(torch.mean((est[:,0] - beta0)**2 + (est[:,1] - B0)**2))