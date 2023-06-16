# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:09:49 2023

@author: LaurentRoberge
"""

import time
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import PriorityFloodFlowRouter, SpaceLargeScaleEroder

from concentration_tracker_SPACE import ConcentrationTrackerSPACE

# %% Filter warnings
warnings.filterwarnings('ignore')

# %% Set up: grid, time, and other parameters

C_initial = 1
C_br = 0
P = 0
D = 0

nrows = 3
ncols = 50
dx = 10
dy = dx
dt = 1
total_t = 500
U = 0.001

H_init = 0.1
K_sed = 0.02
K_br = 0.01
m_sp = 0.5
n_sp = 1.0
F_f = 0.0
phi = 0.0
H_star = 0.1
v_s = 0.1

grid_seed = 7

# Calculated from user inputs
n_core_nodes = (nrows-2)*(ncols-2)
total_t = total_t + dt
ndt = int(total_t // dt)
uplift_per_step = U * dt
outlet_node = ncols
node_next_to_outlet = ncols+1


# %% C-F-L Condition (FastScape)
# Δt <= Cmax * (Δx / (K * (A ** m)))

area = (ncols * dx) * (nrows * dx)

dt_max = 1 * (dx / (max(K_br,K_sed) * (area ** m_sp)))
if dt > dt_max:
    raise Exception(str("Timestep length " + str(dt) + " is longer than C-F-L condition allows. Model may be unstable."))


# %% Generate model grid from initial noise

mg = RasterModelGrid((nrows, ncols), dx)
mg.axis_units = ('m', 'm')
mg.set_status_at_node_on_edges(right=4,
                               top=4,
                               left=4,
                               bottom=4)
mg.status_at_node[outlet_node] = mg.BC_NODE_IS_FIXED_VALUE

# Soil depth
_ = mg.add_zeros('soil__depth', at='node', units= ['m','m'])
mg.at_node['soil__depth'] += H_init
mg.at_node['soil__depth'][outlet_node] = 0

# Bedrock elevation
np.random.seed(grid_seed)
_ = mg.add_zeros('bedrock__elevation', at='node', units= ['m','m'])
mg.at_node['bedrock__elevation'] += mg.node_x / 1000
# mg.at_node['bedrock__elevation'] += np.random.rand(mg.number_of_nodes) / 10
mg.at_node['bedrock__elevation'][outlet_node] = 0

# Topographic elevation
_ = mg.add_zeros('topographic__elevation', at='node', units= ['m','m'])
mg.at_node['topographic__elevation'][:] += mg.at_node['bedrock__elevation']
mg.at_node['topographic__elevation'][:] += mg.at_node['soil__depth']

# # Magnetic susceptibility concentration field
C = mg.add_zeros('sed_property__concentration', at='node', units= ['kg/m^3','kg/m^3'])
mg.at_node['sed_property__concentration'] += 0#C_initial
# mg.at_node['sed_property__concentration'][mg.node_x > ncols/2] += C_initial
mg.at_node['sed_property__concentration'][ncols + int(3*ncols/4)-1] += C_initial

n_core_nodes = len(mg.core_nodes)
core_ids = np.append(outlet_node, mg.core_nodes)
    
# %% Instantiate model components
fr = PriorityFloodFlowRouter(mg)
fr.run_one_step()

sp = SpaceLargeScaleEroder(mg,
                           K_sed=K_sed,
                           K_br=K_br,
                           F_f=F_f,
                           phi=phi,
                           H_star=H_star,
                           m_sp=m_sp,
                           n_sp=n_sp,
                           v_s=v_s
                           )

ctSP = ConcentrationTrackerSPACE(mg,
                                 sp,
                                 concentration_initial=C_initial,
                                 concentration_in_bedrock=C_br,
                                 local_production_rate=P,
                                 local_decay_rate=D
                                 )

#%% Create empty arrays to fill during loop

T = np.zeros(ndt)       # Time
C_sw_outlet = np.zeros(ndt) # concentration entering outlet node
C_on_grid = np.zeros(ndt)
C_in_flux = np.zeros(ndt)
topo_total = np.zeros(ndt)
br_total = np.zeros(ndt)
soil_total = np.zeros(ndt)
QC_out = np.zeros(ndt)
Q_out = np.zeros(ndt)
Vol_removed = np.zeros(ndt)
C_removed = np.zeros(ndt)

E_sed = np.zeros([ndt,len(core_ids)])    
E_br = np.zeros([ndt,len(core_ids)])    
E_total = np.zeros([ndt,len(core_ids)])   

ymax = 2.5

# Set colour maps for plotting
cmap_terrain = mpl.cm.get_cmap("terrain").copy()
cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
cmap_Sm = mpl.cm.get_cmap("Blues").copy()

def plot_channel_profile():
        
    distance = mg.node_x[core_ids]
    elev_topo = mg.at_node['topographic__elevation'][core_ids]
    elev_br = mg.at_node['bedrock__elevation'][core_ids]
    C_soil = mg.at_node['sed_property__concentration'][core_ids]
    
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    plt.figure()    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,6)
    
    ax1.set_xlabel('Distance (m)', fontsize=18, color='black')
    ax1.set_ylabel('Elevation (m)', fontsize=18, color='black')
    plt.plot(distance,elev_topo,distance,elev_br)
    plt.ylim(0,ymax)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(['Soil elevation','Bedrock elevation'],loc="upper left")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sm Concentration (kg/m^3)', fontsize=18, color='black')
    plt.ylim(0,1)
    plt.plot(distance,C_soil,color='black',label="Sm Concentration")
    ax2.tick_params(axis='y', labelcolor='black')    
    ax2.legend(loc="upper right")
        
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()  

# %% Plot initial condition

# imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink')
# plt.show()

plot_channel_profile()

# %% Model Run

# plot_channel_profile()

# Set elapsed model time to 0 years
elapsed_time = 0

# Set initial timestamp (used to print progress updates)
start_time = time.time()

for i in range(ndt):
    # Update time counter
    elapsed_time = i*dt
    
    # Old topo and concentration
    topo_old = mg.at_node['topographic__elevation'][mg.core_nodes].copy()
    C_old = mg.at_node['sed_property__concentration'][mg.core_nodes].copy()
    
    # Add uplift
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    
    # Update topographic elevation to match bedrock and soil depths
    mg.at_node['topographic__elevation'][:] = (mg.at_node["bedrock__elevation"]
                                               + mg.at_node["soil__depth"])
    
    # Run DepthDependentDiffuser
    fr.run_one_step()
    
    # Calculate concentration
    sp.run_one_step(dt=dt)
    
    ctSP.run_one_step(dt=dt)
    
    # plot
    if i*dt % 50 == 0:
        
        plot_channel_profile()
    
    # New topo and concentration
    topo_new = mg.at_node['topographic__elevation'][mg.core_nodes].copy()    
    C_new = mg.at_node['sed_property__concentration'][mg.core_nodes].copy()    
    
    # Collect stats
    
    T[i] = elapsed_time
    E_sed[i,:] = dt*sp.Es[core_ids]
    E_br[i,:] = dt*sp.Er[core_ids]
    E_total[i,:] = E_sed[i,:] + E_br[i,:]
    
    C_sw_outlet[i] = ctSP._C_sw[node_next_to_outlet]
    # topo_total[i] = dx*dy*np.sum(mg.at_node['topographic__elevation'][core_ids])-(i*dx*dy*n_core_nodes*uplift_per_step)
    # br_total[i] = dx*dy*np.sum(mg.at_node['bedrock__elevation'][core_ids])-(i*dx*dy*n_core_nodes*uplift_per_step)
    # soil_total[i] = dx*dy*np.sum(mg.at_node['soil__depth'][core_ids])
    
    C_on_grid[i] = np.sum(mg.at_node['sed_property__concentration'][mg.core_nodes]
                        * mg.at_node['soil__depth'][mg.core_nodes])*dx*dy
    QC_out[i] = ctSP._QsCsw_out[node_next_to_outlet]
    QC_out2[i] = QC_out[i-1]+ ctSP._QsCsw_out[node_next_to_outlet]
    Q_out[i] = mg.at_node['sediment__outflux'][node_next_to_outlet]*dt
    C_in_flux[i] = np.sum(ctSP._QsCsw_in[mg.core_nodes])
    Vol_removed[i] = (dx*dy*np.sum(topo_old)
                      - dx*dy*np.sum(topo_new - uplift_per_step)
                      )
    C_removed[i] = dx*dy*np.sum(C_old) - dx*dy*np.sum(C_new)
    Vol_balance = Vol_removed - Q_out
    C_balance = C_removed - (QC_out + C_in_flux)
    C_balance2 = C_on_grid + QC_out
    
# %%

plt.figure()
plt.plot(T,C_sw_outlet,label="Concentration in water column entering outlet")
plt.legend()
plt.show()

plt.figure()
plt.plot(T,C_balance,label="Concentration balance on grid")
plt.plot(T,Vol_balance,label="Volume balance on grid")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(T,topo_total,label="Total topographic volume on grid")
# plt.plot(T,br_total,label="Total bedrock volume on grid")
# plt.plot(T,soil_total,label="Total soil volume on grid")
# plt.legend()
# plt.show()

# plt.figure()
# #plt.plot(mg.node_x[mg.core_nodes],D_sw[mg.core_nodes],label="Deposition")
# plt.plot(mg.node_x[mg.core_nodes],sp.Es[mg.core_nodes],label="Erosion (sed)")
# plt.plot(mg.node_x[mg.core_nodes],sp.Er[mg.core_nodes],label="Erosion (br)")
# plt.legend()
# plt.show()

# E_tot = E_r+E_s

# plt.figure()
# plt.plot(mg.node_x[mg.core_nodes],D_sw[mg.core_nodes],label="Deposition")
# plt.plot(mg.node_x[mg.core_nodes],E_tot[mg.core_nodes],label="Erosion (total)")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(mg.node_x[mg.core_nodes],D_sw[mg.core_nodes]/E_tot[mg.core_nodes],label="Deposition/Erosion")
# plt.legend()
# plt.show()