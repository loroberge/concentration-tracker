# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:37:40 2023

@author: LaurentRoberge
"""

import numpy as np

def create_C_H_arrays_1Node(ndt, C_initial, H_initial):
    C = np.zeros(ndt+1)
    C[0] = C_initial
    H = np.zeros(ndt+1)
    H[0] = H_initial
    
    return C, H

def create_C_H_arrays_MultiNode(ndt, nnodes, C_initial, H_initial):
    C = np.zeros([nnodes,ndt+1])
    H = np.zeros([nnodes,ndt+1])
    for n in range(nnodes):
        C[n,0] = C_initial[n]
        H[n,0] = H_initial[n]
    
    return C, H
    
def create_empty_records_1Node(ndt, C_in, C_out, q_in, q_out):
    records = {'C_in' : np.zeros(ndt+1),
               'C_out' : np.zeros(ndt+1),
               'q_in' : np.zeros(ndt+1),
               'q_out' : np.zeros(ndt+1),
               'C_local' : np.zeros(ndt+1),
               'dFdx' : np.zeros(ndt+1),
               'C_fluxed' : np.zeros(ndt+1),
               'Production' : np.zeros(ndt+1),
               'Decay' : np.zeros(ndt+1),
               'M_in' : np.zeros(ndt+1),
               'M_out' : np.zeros(ndt+1),
               }

    records['C_in'][0] = C_in
    records['C_out'][0] = C_out
    records['q_in'][0] = q_in
    records['q_out'][0] = q_out
    records['C_local'][0] = np.nan
    records['dFdx'][0] = np.nan
    records['C_fluxed'][0] = np.nan
    records['Production'][0] = np.nan
    records['Decay'][0] = np.nan
    records['M_in'][0] = C_in*q_in
    records['M_out'][0] = C_out*q_out
    
    return records

def create_empty_records_MultiNode(ndt, nnodes, C_links, q, dy, dt):
    nlinks = nnodes+1
    
    records = {'C_links' : np.zeros([nlinks,ndt+1]),
               'q' : np.zeros([nlinks,ndt+1]),
               'C_local' : np.zeros([nnodes,ndt+1]),
               'dFdx' : np.zeros([nnodes,ndt+1]),
               'C_fluxed' : np.zeros([nnodes,ndt+1]),
               'Production' : np.zeros([nnodes,ndt+1]),
               'Decay' : np.zeros([nnodes,ndt+1]),
               'M_links' : np.zeros([nlinks,ndt+1])
               }

    records['C_links'][:,0] = C_links
    records['q'][:,0] = q
    records['C_local'][:,0] = np.nan
    records['dFdx'][:,0] = np.nan
    records['C_fluxed'][:,0] = np.nan
    records['Production'][:,0] = np.nan
    records['Decay'][:,0] = np.nan
    records['M_links'][:,0] = C_links*q*dy*dt
    
    return records
    
def record_data_1Node(records, t, dt, dy, C_in, C_out, q_in, q_out, C_local, dFdx, 
                Production, Decay):
    records['C_in'][t] = C_in
    records['C_out'][t] = C_out
    records['q_in'][t] = q_in
    records['q_out'][t] = q_out
    records['C_local'][t] = C_local
    records['dFdx'][t] = dFdx
    records['C_fluxed'][t] = dt * (- dFdx)
    records['Production'][t] = Production
    records['Decay'][t] = Decay
    records['M_in'][t] = C_in*q_in*dy*dt
    records['M_out'][t] = C_out*q_out*dy*dt
    
    return records
    
def record_data_MultiNode(records, n, t, dt, dy, C_links, q, C_local, dFdx, 
                          Production, Decay):
    records['C_links'][n,t] = C_links[n]
    records['C_links'][n+1,t] = C_links[n+1]
    records['q'][n,t] = q[n]
    records['q'][n+1,t] = q[n+1]
    records['C_local'][n,t] = C_local
    records['dFdx'][n,t] = dFdx
    records['C_fluxed'][n,t] = dt * (- dFdx)
    records['Production'][n,t] = Production
    records['Decay'][n,t] = Decay
    records['M_links'][n,t] = C_links[n]*q[n]*dy*dt
    records['M_links'][n+1,t] = C_links[n+1]*q[n+1]*dy*dt
    
    return records