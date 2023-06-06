# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:20:10 2023

@author: LaurentRoberge
"""

# %% Import
import numpy as np
from library import (create_empty_records_1Node,
                     create_C_H_arrays_1Node,
                     record_data_1Node
                     )
from library_plot import (plot_H, 
                          plot_C,
                          plot_Cin_Cout,
                          plot_qin_qout,
                          plot_Min_Mout,
                          plot_Clocal,
                          plot_dFdx,
                          plot_Cfluxed,
                          plot_P,        
                          plot_D,
                          plot_C_all
                          )

# %% Define parameters

# Defined by user

full_mixing = False

t_total = 10
dt = 0.1
dx = 1
dy = 1

C_initial = 0.5
H_initial = 1
q_in = 1
C_in = 0.1
q_out = 1

P = 0
D = 0

# Calculated from above
ndt = round(t_total//dt) # number of timesteps
t_plot = np.linspace(0,ndt*dt,ndt+1)
C_out = C_initial
[C, H] = create_C_H_arrays_1Node(ndt, C_initial, H_initial)
records = create_empty_records_1Node(ndt, C_in, C_out, q_in, q_out)

# %% Test equation

for t in range(1,ndt+1):
    
    H[t] = H[t-1] + (q_in - q_out)*dt/(dx*dy)
    
    if H[t] > 0:

        C_local = C[t-1] * (H[t-1]/H[t])
        Production = (dt*P/2) * (1 + H[t-1]/H[t])
        Decay = (dt*D/2) * (1 + H[t-1]/H[t])
        
        if full_mixing:
                
            if q_out<=q_in:
                C_out = C_in
            else:
                C_out = C_in*(q_in/q_out) + C[t-1]*(1-q_in/q_out)
        
        dFdx = (q_out * C_out - q_in * C_in)/dx
        
        C[t] = C_local + (dt/H[t]) * (- dFdx) + Production - Decay
        
        if not full_mixing:
            C_out = C[t]
        
    else:
        H[t] = 0
                
        C_out = C_in # this means any sediment fluxed to the next node has C from 2 nodes up (non-local transport)
        # C_out = C_bedrock # this is for the situation in which any sediment fluxed to the next node is bedrock acquired from this node
    
    records = record_data_1Node(records, t, dt, dy, C_in, C_out, q_in, q_out, 
                          C_local, dFdx, Production, Decay
                          )

#%% Plot

plot_H(t_plot, H)
plot_C(t_plot, C)
# plot_Cin_Cout(t_plot, records['C_in'], records['C_out'])
# plot_qin_qout(t_plot, records['q_in'], records['q_out'])
# plot_Min_Mout(t_plot, records['M_in'], records['M_out'])
# plot_Clocal(t_plot, records['C_local'])
# plot_dFdx(t_plot, records['dFdx'])
# plot_Cfluxed(t_plot, records['C_fluxed'])
# plot_P(t_plot, records['Production'])
# plot_D(t_plot, records['Decay'])
plot_C_all(t_plot, records['C_local'], records['C_fluxed'], 
            records['Production'], records['Decay'])
     
# %% Mass balance check over entire model runtime  

M_in_total = np.sum(records['M_in'][:-1])
M_out_total = np.sum(records['M_out'][1:])
M_start = np.sum(C[0] * H[0])
M_end = np.sum(C[-1] * H[-1])
M_balance = M_start + M_in_total - M_end - M_out_total

print("Total mass in: " + str(M_in_total))
print("Total mass out: " + str(M_out_total))
print("Mass at start: " + str(M_start))
print("Mass at end: " + str(M_end))
print("Mass balance: " + str(M_balance))

