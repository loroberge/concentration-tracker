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

# %% Set up: grid, time, and other parameters
nrows = 3
ncols = 20
dx = 50
dt = 10
total_t = 1000
U = 0.001

H_init = 0.1
K_sed = 0.02
K_br = 0.01
m_sp = 0.5
n_sp = 1.0
F_f = 0.0
phi = 0.0
H_star = 0.1

grid_seed = 7

# Calculated from user inputs
n_core_nodes = (nrows-2)*(ncols-2)
total_t = total_t + dt
ndt = int(total_t // dt)
uplift_per_step = U * dt
outlet_node = ncols
node_next_to_outlet = ncols+1

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

# Bedrock elevation
np.random.seed(grid_seed)
_ = mg.add_zeros('bedrock__elevation', at='node', units= ['m','m'])
mg.at_node['bedrock__elevation'] += np.random.rand(mg.number_of_nodes) / 10
mg.at_node['bedrock__elevation'][outlet_node] = 0

# Topographic elevation
_ = mg.add_zeros('topographic__elevation', at='node', units= ['m','m'])
mg.at_node['topographic__elevation'][:] += mg.at_node['bedrock__elevation']
mg.at_node['topographic__elevation'][:] += mg.at_node['soil__depth']
    
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
                           )

#%% Create empty arrays to fill during loop

T = np.zeros(ndt)       # Time
E_sed = np.zeros([ndt,len(core_ids)])    
E_br = np.zeros([ndt,len(core_ids)])    
E_total = np.zeros([ndt,len(core_ids)])    

ymax = 1.5

# Set colour maps for plotting
cmap_terrain = mpl.cm.get_cmap("terrain").copy()
cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
cmap_Sm = mpl.cm.get_cmap("Blues").copy()

def plot_channel_profile():
        
    distance = mg.node_x[core_ids]
    elev_topo = mg.at_node['topographic__elevation'][core_ids]
    elev_br = mg.at_node['bedrock__elevation'][core_ids]
#    C_soil = mg.at_node['sed_property__concentration'][core_ids]
    
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
    
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Sm Concentration (kg/m^3)', fontsize=18, color='black')
    # plt.ylim(0,1)
    # plt.plot(distance,C_soil,color='black',label="Sm Concentration")
    # ax2.tick_params(axis='y', labelcolor='black')    
    # ax2.legend(loc="upper right")
        
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()  

# %% Plot initial hillslope

imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
plt.show()
imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
plt.show()
imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
plt.show()
# imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink')
# plt.show()

# %% Model Run

#plot_hill_profile()

# Set elapsed model time to 0 years
elapsed_time = 0

# Set initial timestamp (used to print progress updates)
start_time = time.time()

for i in range(ndt):
    # Update time counter
    elapsed_time = i*dt
    
    # Add uplift
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    
    # Update topographic elevation to match bedrock and soil depths
    mg.at_node['topographic__elevation'][:] = (mg.at_node["bedrock__elevation"]
                                               + mg.at_node["soil__depth"])
    
    # Run DepthDependentDiffuser
    fr.run_one_step()
    
    # Calculate concentration
    vol, vol2 = sp.run_one_step(dt=dt)
    
    T[i] = elapsed_time
    E_sed[i,:] = dt*sp.Es[core_ids]
    E_br[i,:] = dt*sp.Er[core_ids]
    E_total[i,:] = E_sed[i,:] + E_br[i,:]
    
    # plot_hill_profile()

    if i*dt % 50 == 0:
        
        plot_channel_profile()
        
        
# %%        
        # cell_area = dx*dx
        # qs_out = ((sp.qs_in[core_ids] 
        #            + sp.Es[core_ids]*cell_area 
        #            + (1.0-F_f)*sp.Er[core_ids]*cell_area)
        #            / (1.0+(sp.v_s*cell_area/sp.q[core_ids()]))
        #            )
        
        
        # D_rate = sp._v_s*qs_out/sp.q[core_ids]
        
        