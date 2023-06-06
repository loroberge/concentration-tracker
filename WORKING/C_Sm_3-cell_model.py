# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:20:52 2023

@author: LaurentRoberge
"""


# %% Import
import numpy as np
from library import (create_empty_records_MultiNode,
                     create_C_H_arrays_MultiNode,
                     record_data_MultiNode
                     )
from library_plot import (plot_H_MultiNode,
                          plot_C_MultiNode,
                          plot_C_links_MultiNode,
                          plot_q_MultiNode,
                          plot_Min_Mout_MultiNode,
                          plot_Clocal_MultiNode,
                          plot_dFdx_MultiNode,
                          plot_Cfluxed_MultiNode,
                          plot_P_MultiNode,
                          plot_D_MultiNode,
                          plot_C_all_MultiNode
                          )

# %% Define number of nodes

nnodes = 3

# Calculated from above
nlinks = nnodes+1
C_initial = np.zeros(nnodes)
H_initial = np.zeros(nnodes)
q = np.zeros(nlinks)
C_links = np.zeros(nlinks)
P = np.zeros(nnodes)
D = np.zeros(nnodes)

# %% Define parameters
# Defined by user

full_mixing = False

t_total = 10
dt = 0.1
dx = 1
dy = 1

C_links[0] = 0.1 # Concentration fluxing into node 1
C_initial += [0.5, 0.5, 0.5]
H_initial += [1, 1, 1]

q += [1, 1, 1, 1]
P += [0, 0, 0]
D += [0, 0, 0]

# Calculated from above
ndt = round(t_total//dt) # number of timesteps
t_plot = np.linspace(0,ndt*dt,ndt+1)

C_links[1] = C_initial[0] # Concentration fluxing between nodes 1 & 2
C_links[2] = C_initial[1] # Concentration fluxing out of node 2
C_links[3] = C_initial[2] # Concentration fluxing out of node 2

[C, H] = create_C_H_arrays_MultiNode(ndt, nnodes, C_initial, H_initial)
records = create_empty_records_MultiNode(ndt, nnodes, C_links, q, dy, dt)

# %% Test equation

for t in range(1,ndt+1):
    
    for n in range(nnodes):
        
        H[n,t] = H[n,t-1] + (q[n] - q[n+1])*dt/(dx*dy)
        
        if H[n,t] > 0:
    
            C_local = C[n,t-1] * (H[n,t-1]/H[n,t])
            Production = (dt*P[n]/2) * (1 + H[n,t-1]/H[n,t])
            Decay = (dt*D[n]/2) * (1 + H[n,t-1]/H[n,t])
            
            if full_mixing:
                
                if q[n+1]<=q[n]:
                    C_links[n+1] = C_links[n]
                else:
                    C_links[n+1] = C_links[n]*(q[n]/q[n+1]) + C[n,t-1]*(1-q[n]/q[n+1])
            
            dFdx = (q[n+1] * C_links[n+1] - q[n] * C_links[n])/dx
            
            C[n,t] = C_local + (dt/H[n,t]) * (- dFdx) + Production - Decay
            
            if not full_mixing:
                C_links[n+1] = C[n,t]
            
        else:
            H[n,t] = 0
                    
            C_links[n+1] = C_links[n] # this means any sediment fluxed to the next node has C from 2 nodes up (non-local transport)
            # C_out = C_bedrock # this is for the situation in which any sediment fluxed to the next node is bedrock acquired from this node
        
        records = record_data_MultiNode(records, n, t, dt, dy, C_links, q, 
                                        C_local, dFdx, Production, Decay
                                        )

#%% Plot

plot_H_MultiNode(nnodes, t_plot, H)
plot_C_MultiNode(nnodes, t_plot, C)
# plot_C_links_MultiNode(nnodes, t_plot, records)
# plot_q_MultiNode(nnodes, t_plot, records)
# plot_Min_Mout_MultiNode(nnodes, t_plot, records)
# plot_Clocal_MultiNode(nnodes, t_plot, records)
# plot_dFdx_MultiNode(nnodes, t_plot, records)
# plot_Cfluxed_MultiNode(nnodes, t_plot, records)
# plot_P_MultiNode(nnodes, t_plot, records)
# plot_D_MultiNode(nnodes, t_plot, records)
# plot_C_all_MultiNode(nnodes, t_plot, records)
     
M_in_total = np.sum(records['M_links'][0,:-1])
M_out_total = np.sum(records['M_links'][-1,1:])
M_start = np.sum(C[:,0] * H[:,0])
M_end = np.sum(C[:,-1] * H[:,-1])
M_balance = M_start + M_in_total - M_end - M_out_total

print("Total mass in: " + str(M_in_total))
print("Total mass out: " + str(M_out_total))
print("Mass at start: " + str(M_start))
print("Mass at end: " + str(M_end))
print("Mass balance: " + str(M_balance))

