# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:26:14 2023

@author: LaurentRoberge
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_H(t_plot, H):
    
    plt.figure()
    plt.plot(t_plot,H)
    plt.title('H through time')
    plt.xlabel('Time (y)')
    plt.ylabel('Sediment thickness (m)')
    plt.show()

def plot_C(t_plot, C):
    
    plt.figure()
    plt.plot(t_plot,C)
    plt.title('C through time')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration of magnetic particles (kg/m^3)')
    plt.show()

def plot_Cin_Cout(t_plot, record_C_in, record_C_out):
        
    plt.figure()
    plt.plot(t_plot,record_C_in,t_plot,record_C_out)
    plt.title('C_in and C_out')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration')
    plt.legend(['C_in','C_out'])
    plt.show()

def plot_qin_qout(t_plot, record_q_in, record_q_out):
        
    plt.figure()
    plt.plot(t_plot,record_q_in,t_plot,record_q_out)
    plt.title('q_in and q_out')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration')
    plt.legend(['q_in','q_out'])

def plot_Min_Mout(t_plot, record_M_in, record_M_out):
        
    plt.figure()
    plt.plot(t_plot,record_M_in,t_plot,record_M_out)
    plt.title('M_in and M_out')
    plt.xlabel('Time (y)')
    plt.ylabel('Mass')
    plt.legend(['M_in','M_out'])
    plt.show()

def plot_Clocal(t_plot, record_C_local):
        
    plt.figure()
    plt.plot(t_plot,record_C_local)
    plt.title('C_local through time')
    plt.xlabel('Time (y)')
    plt.ylabel('Local concentration factor')
    plt.show()

def plot_dFdx(t_plot, record_dFdx):
    
    plt.figure()
    plt.plot(t_plot,record_dFdx)
    plt.title('Change in dFdx with time')
    plt.xlabel('Time (y)')
    plt.ylabel('dFdx')
    plt.show()

def plot_Cfluxed(t_plot, record_C_fluxed):
        
    plt.figure()
    plt.plot(t_plot,record_C_fluxed)
    plt.title('Concentration fluxed into node')
    plt.xlabel('Time (y)')
    plt.ylabel('C fluxed into node')
    plt.show()

def plot_P(t_plot, record_Production):
        
    plt.figure()
    plt.plot(t_plot,record_Production)
    plt.title('Concentration Produced locally')
    plt.xlabel('Time (y)')
    plt.ylabel('C Produced')
    plt.show()

def plot_D(t_plot, record_Decay):
        
    plt.figure()
    plt.plot(t_plot,record_Decay)
    plt.title('Concentration Decayed locally')
    plt.xlabel('Time (y)')
    plt.ylabel('C decayed')
    plt.show()

def plot_C_all(t_plot, record_C_local, record_C_fluxed, record_Production,
               record_Decay):
        
    plt.figure()
    plt.plot(t_plot,record_C_local,t_plot,record_C_fluxed,t_plot,
             record_Production,t_plot,record_Decay)
    plt.title('C local, fluxed in, produced, and decayed')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration')
    plt.legend(['C local','C fluxed in','C produced','C decayed'])
    plt.show()
    
# Multi-node model ------------------------------------------------------------
def plot_H_MultiNode(nnodes, t_plot, H):
    
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,H[n,:],label='Node #'+str(n+1))
    plt.title('H through time')
    plt.xlabel('Time (y)')
    plt.ylabel('Sediment thickness (m)')
    plt.legend()
    plt.show()

def plot_C_MultiNode(nnodes, t_plot, C):
    
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,C[n,:],label='Node #'+str(n+1))    
    plt.title('C through time')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration of magnetic particles (kg/m^3)')
    plt.legend()
    plt.show()

def plot_C_links_MultiNode(nnodes, t_plot, records):
    
    nlinks = nnodes+1
    
    plt.figure()
    for n in range(nlinks):
        plt.plot(t_plot,records['C_links'][n,:],label='Link #'+str(n+1))
    plt.title('C fluxing between nodes')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()

def plot_q_MultiNode(nnodes, t_plot, records):
       
    nlinks = nnodes+1

    plt.figure()
    for n in range(nlinks):
        plt.plot(t_plot,records['q'][n,:],label='Link #'+str(n+1))
    plt.title('q fluxing between nodes')
    plt.xlabel('Time (y)')
    plt.ylabel('Sediment flux (m3/y)')
    plt.legend()
    plt.show()
    
def plot_Min_Mout_MultiNode(nnodes, t_plot, records):
        
    nlinks = nnodes+1

    plt.figure()
    for n in range(nlinks):
        plt.plot(t_plot,records['M_links'][n,:],label='Link #'+str(n+1))
    plt.title('M fluxing between nodes')
    plt.xlabel('Time (y)')
    plt.ylabel('Mass (kg)')
    plt.legend()
    plt.show()

def plot_Clocal_MultiNode(nnodes, t_plot, records):
        
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,records['C_local'][n,:],label='Node #'+str(n+1))
    plt.title('C_local through time')
    plt.xlabel('Time (y)')
    plt.ylabel('Local concentration factor')
    plt.legend()
    plt.show()

def plot_dFdx_MultiNode(nnodes, t_plot, records):
    
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,records['dFdx'][n,:],label='Node #'+str(n+1))
    plt.title('dFdx through time')
    plt.xlabel('Time (y)')
    plt.ylabel('dFdx')
    plt.legend()
    plt.show()

def plot_Cfluxed_MultiNode(nnodes, t_plot, records):
        
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,records['C_fluxed'][n,:],label='Node #'+str(n+1))
    plt.title('Concentration fluxed into node')
    plt.xlabel('Time (y)')
    plt.ylabel('C fluxed into node')
    plt.legend()
    plt.show()

def plot_P_MultiNode(nnodes, t_plot, records):
        
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,records['Production'][n,:],label='Node #'+str(n+1))
    plt.title('Concentration Produced locally')
    plt.xlabel('Time (y)')
    plt.ylabel('C Produced')
    plt.legend()
    plt.show()

def plot_D_MultiNode(nnodes, t_plot, records):
        
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,records['Decay'][n,:],label='Node #'+str(n+1))
    plt.title('Concentration Decayed locally')
    plt.xlabel('Time (y)')
    plt.ylabel('C decayed')
    plt.legend()
    plt.show()

def plot_C_all_MultiNode(nnodes, t_plot, records):
    
    plt.figure()
    for n in range(nnodes):
        plt.plot(t_plot,records['C_local'][n,:],label='C_local @ Node #'+str(n+1))
        plt.plot(t_plot,records['C_fluxed'][n,:],label='C_fluxed @ Node #'+str(n+1))
        plt.plot(t_plot,records['Production'][n,:],label='P @ Node #'+str(n+1))
        plt.plot(t_plot,records['Decay'][n,:],label='D @ Node #'+str(n+1))        
    plt.title('C local, fluxed in, produced, and decayed')
    plt.xlabel('Time (y)')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()
    

