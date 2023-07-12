# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:48:11 2023

@author: LaurentRoberge
"""

# %% Import

import time
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import (ExponentialWeatherer,
                                DepthDependentDiffuser,
                                PriorityFloodFlowRouter,
                                SpaceLargeScaleEroder
                                )

from concentration_tracker_DDD import ConcentrationTrackerDDD
from concentration_tracker_SPACE import ConcentrationTrackerSPACE

# %% Filter warnings
warnings.filterwarnings('ignore')

# %% Define input parameters

# Set soil production rate to zero (to avoid C difference between bedrock and soil)
soil_production_maximum_rate = 0.001
soil_production_decay_depth = 1

nrows = 30
ncols = 30
dx = 50
dy = dx
dt = 1

total_t = 5000
U = 0.01

C_initial = 1
C_br = 0
P = 0
D = 0

H_init = 0.1
K_sed = 0.01
K_br = 0.005
m_sp = 0.5
n_sp = 1.0
F_f = 0.0
phi = 0.0
H_star = 0.5
v_s = 1

grid_seed = 11

# Calculated from user inputs
n_core_nodes = (nrows-2)*(ncols-2)
total_t = total_t + dt
ndt = int(total_t // dt)
uplift_per_step = U * dt
outlet_node = 0
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
mg.at_node['soil__depth'] += 5 #mg.node_x/50 
#mg.at_node['soil__depth'][ncols + int(3*ncols/4)-1] += 2 
mg.at_node['soil__depth'][outlet_node] = 0

# Bedrock elevation
_ = mg.add_zeros('bedrock__elevation', at='node', units= ['m','m'])
mg.at_node['bedrock__elevation'] += np.random.rand(mg.number_of_nodes) / 100
mg.at_node['bedrock__elevation'] += (mg.node_x + mg.node_y)/1000000
# mg.at_node['bedrock__elevation'][mg.node_x > dx*ncols/2] += 8
mg.at_node['bedrock__elevation'][outlet_node] = 0

# Topographic elevation
_ = mg.add_zeros('topographic__elevation', at='node', units= ['m','m'])
mg.at_node['topographic__elevation'][:] += mg.at_node['bedrock__elevation']
mg.at_node['topographic__elevation'][:] += mg.at_node['soil__depth']

# Magnetic susceptibility concentration field
C = mg.add_zeros('sed_property__concentration', at='node', units= ['kg/m^3','kg/m^3'])
mg.at_node['sed_property__concentration'] += 0 #C_initial
# mg.at_node['sed_property__concentration'][mg.node_x > ncols/2] += C_initial
# mg.at_node['sed_property__concentration'][ncols - int(3*ncols/4)-1] += C_initial
mg.at_node['sed_property__concentration'][530] += C_initial
mg.at_node['sed_property__concentration'][320] += C_initial
mg.at_node['sed_property__concentration'][610] += C_initial

core_ids = np.append(outlet_node, mg.core_nodes)

# %% C-F-L Condition (FastScape)
# Δt <= Cmax * (Δx / (K * (A ** m)))

area = (ncols * dx) * (nrows * dx)

dt_max = 1 * (dx / (max(K_br,K_sed) * (area ** m_sp)))
if dt > dt_max:
    raise Exception(str("Timestep length " + str(dt) + " is longer than C-F-L condition allows. Model may be unstable."))

# %% Instantiate model components
fr = PriorityFloodFlowRouter(mg)
fr.run_one_step()

ew = ExponentialWeatherer(mg,
                          soil_production_maximum_rate=soil_production_maximum_rate,
                          soil_production_decay_depth=soil_production_decay_depth)

ddd = DepthDependentDiffuser(mg)

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

ctDDD = ConcentrationTrackerDDD(mg,
                             concentration_initial=C,
                             concentration_in_bedrock=C_br,
                             local_production_rate=P,
                             local_decay_rate=D,
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

ymax = np.max(mg.at_node["topographic__elevation"][core_ids])+0.5

# Set colour maps for plotting
cmap_terrain = mpl.cm.get_cmap("terrain").copy()
cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
cmap_Sm = mpl.cm.get_cmap("Blues").copy()

def plot_hill_profile():
        
    distance = mg.node_x[core_ids]
    elev_topo = mg.at_node['topographic__elevation'][core_ids]
    elev_br = mg.at_node['bedrock__elevation'][core_ids]
    H_soil = mg.at_node['soil__depth'][core_ids]
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
    ax1.legend(['Soil depth','Concentration'],loc="upper left")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sm Concentration (kg/m^3)', fontsize=18, color='black')
    plt.ylim(0,1)
    plt.plot(distance,C_soil,color='black',label="Sm Concentration")
    ax2.tick_params(axis='y', labelcolor='black')    
    ax2.legend(loc="upper right")
        
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()  

# %% Plot initial hillslope

imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
plt.show()
imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
plt.show()
imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
plt.show()
imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink')
plt.show()

# plot_hill_profile()

# %% Model Run

# plot_hill_profile()

# Set elapsed model time to 0 years
elapsed_time = 0

# Set initial timestamp (used to print progress updates)
start_time = time.time()

for i in range(ndt):
    # Update time counter
    elapsed_time = i*dt
    
    # Add uplift
    # mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    
    # Run ExponentialWeatherer
    ew.calc_soil_prod_rate()
    soil_prod_per_step = mg.at_node['soil_production__rate'][mg.core_nodes] * dt
    
    # Convert bedrock to soil using soil production rate
    mg.at_node['bedrock__elevation'][mg.core_nodes] -= soil_prod_per_step
    mg.at_node['soil__depth'][mg.core_nodes] += soil_prod_per_step
    
    # Update topographic elevation to match bedrock and soil depths
    mg.at_node['topographic__elevation'][:] = (mg.at_node["bedrock__elevation"]
                                               + mg.at_node["soil__depth"])
    
    # Run Flow Router
    fr.run_one_step()
    
    # Run SPACE
    sp.run_one_step(dt=dt)
    
    # Calculate concentration
    ctSP.run_one_step(dt=dt)
    
    # Run DepthDependentDiffuser
    ddd.run_one_step(dt=dt)
    
    # Calculate concentration
    ctDDD.run_one_step(dt=dt)
    
    T[i] = elapsed_time
    
    # plot_hill_profile()

    if i*dt % 100 == 0:
        
        # plot_hill_profile()
        
        imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink', limits=[0,0.1])
        plt.show()
        imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
        plt.show()
        imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
        plt.show()
        imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
        plt.show()

# %% Plot final hillslope

# imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
# plt.show()




