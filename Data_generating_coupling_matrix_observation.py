import numpy as np
import math
import networkx as nx

## Generating a graph and its adjacency matrix

# An irregular random graph
n = 500
M_n = n**(0.3)
epsilon = n**(-0.1)
d = int( (2*M_n) / epsilon**2 )
E = n*d / 4 # number of edges
G = nx.random_regular_graph(d, int(n/2), seed=1234)
scaling = n / (2*E)

adjm = nx.adjacency_matrix(G)
A_sub = adjm.todense()
A = np.zeros((n,n))
A[:int(n/2), :int(n/2)] = A_sub
A = scaling*A # Scaled adjacency matrix
# np.savetxt('A_IRG_d44_n500.txt', A, delimiter=',') # Save the coupling matrix


## Comment out below if you want to generate a regular graph and its coupling matrix

# # A regular random graph
# n = 100
# d = 20
# G = nx.random_regular_graph(d, n, seed=1234)
# E = G.number_of_edges() # number of edges
# scaling = n / (2*E)

# adjm = nx.adjacency_matrix(G)
# A = adjm.todense()
# A = scaling*A # Scaled adjacency matrix
# # np.savetxt('A_RG_d20_n100.txt', A, delimiter=',')  # Save the coupling matrix

####################################################################################


## Generating observed binary vectors
def write_txt(list, fname, sep):
    file = open(fname, 'w')
    vstr = ''
    
    for a in list:
        for b in a:
            vstr = vstr + str(b) + sep
        vstr = vstr.rstrip(sep)
        vstr = vstr + '\n'
    
    file.writelines(vstr)
    file.close()
    print('[Saving a file is done]')    

# True parameters
beta= 0.2
B= 0.2

xs_rep= []
R= 50
for r in range(R):
    np.random.seed(r)
    x_old = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])
    
    accept = []
    it= 1000000 
    for i in range(it):
        E_old = 0.5*beta*np.dot(np.dot(x_old, A),x_old) + B*np.sum(x_old)
        
        x_new = x_old.copy()
        random_int = np.random.randint(n, size=1) # random integer less than n
        x_new[random_int] = -x_new[random_int]
        E_new = 0.5*beta*np.dot(np.dot(x_new, A),x_new) + B*np.sum(x_new)
        
        Delta_E = E_new-E_old
        
        if Delta_E > 0:
            accept.append(x_new)
            x_old = x_new.copy()
        else:
            prob = math.exp(Delta_E)
            if prob > np.random.uniform():
                accept.append(x_new)
                x_old = x_new.copy()    
    
    xs_rep.append(accept[-1])

# # Save the generated x's
# write_txt(xs_rep, 'x_IRG_d44_n500_beta2_B2.txt', sep=',')


