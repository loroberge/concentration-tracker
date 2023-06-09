# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:20:00 2023

@author: LaurentRoberge
"""

# %% Import

import time
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import ExponentialWeatherer, DepthDependentDiffuser

from concentration_tracker_DDD import ConcentrationTrackerDDD

# %% Filter warnings
warnings.filterwarnings('ignore')

# %% Define input parameters

# Set soil production rate to zero (to avoid C difference between bedrock and soil)
soil_production_maximum_rate = 0.001
soil_production_decay_depth = 0.00001

nrows = 3
ncols = 50
n_core_nodes = (nrows-2)*(ncols-2)
hill_bottom_node = ncols
hill_bottom_link = (ncols*2)-1

dx = 10
dy = dx
dt = 5

total_t = 15000
ndt = int(total_t // dt)

C_initial = 0.75
C_br = 0

P = 0
D = 0

record_M_total_nodes = np.zeros(ndt)
record_M_total_links = np.zeros(ndt)
record_M_out = np.zeros(ndt)
record_Soil_flux = np.zeros(ndt)
record_H_total_nodes = np.zeros(ndt)

# %% Generate model grid from initial noise

mg = RasterModelGrid((nrows, ncols), dx)
mg.axis_units = ('m', 'm')
mg.set_status_at_node_on_edges(right=4,
                               top=4,
                               left=4,
                               bottom=4)
mg.status_at_node[hill_bottom_node] = mg.BC_NODE_IS_FIXED_VALUE

# Soil depth
_ = mg.add_zeros('soil__depth', at='node', units= ['m','m'])
mg.at_node['soil__depth'] += 1.5 # mg.node_x/4 
#mg.at_node['soil__depth'][ncols + int(3*ncols/4)-1] += 2 

# Bedrock elevation
_ = mg.add_zeros('bedrock__elevation', at='node', units= ['m','m'])
mg.at_node['bedrock__elevation'] += mg.node_x/50
#mg.at_node['bedrock__elevation'][mg.node_x > dx*ncols/2] += 8

# Topographic elevation
_ = mg.add_zeros('topographic__elevation', at='node', units= ['m','m'])
mg.at_node['topographic__elevation'][:] += mg.at_node['bedrock__elevation']
mg.at_node['topographic__elevation'][:] += mg.at_node['soil__depth']

# Magnetic susceptibility concentration field
C = mg.add_zeros('sed_property__concentration', at='node', units= ['kg/m^3','kg/m^3'])
mg.at_node['sed_property__concentration'] += 0#C_initial
# mg.at_node['sed_property__concentration'][mg.node_x > ncols/2] += C_initial
mg.at_node['sed_property__concentration'][ncols + int(3*ncols/4)-1] += C_initial

core_ids = np.append(hill_bottom_node, mg.core_nodes)

# %% Instantiate model components

ew = ExponentialWeatherer(mg,
                          soil_production_maximum_rate=soil_production_maximum_rate,
                          soil_production_decay_depth=soil_production_decay_depth)

ddd = DepthDependentDiffuser(mg)

ctDDD = ConcentrationTrackerDDD(mg,
                             concentration_initial=C,
                             concentration_in_bedrock=C_br,
                             local_production_rate=P,
                             local_decay_rate=D,
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

plot_hill_profile()

# %% Model Run

plot_hill_profile()

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
    
    # Run DepthDependentDiffuser
    ddd.run_one_step(dt=dt)
    
    # Calculate concentration
    ctDDD.run_one_step(dt=dt)
    
    T[i] = elapsed_time
    
    # plot_hill_profile()

    if i*dt % 200 == 0:
        
        plot_hill_profile()
        
        # imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink')
        # plt.show()

    
    record_M_total_nodes[i] = np.sum(mg.at_node['sed_property__concentration'][mg.core_nodes] * 
                                mg.at_node['soil__depth'][mg.core_nodes]
                                )
    record_M_total_links[i] = np.sum(mg.at_link['QC'])
    record_M_out[i] = mg.at_link['QC'][hill_bottom_link]
    record_Soil_flux[i] = mg.at_link['soil__flux'][hill_bottom_link]
    
    record_H_total_nodes[i] = np.sum(mg.at_node['soil__depth'][core_ids])
    #record_M_at_each_node[:,i] = mg.at_node['sed_property__concentration'][core_ids]

# %%
plt.plot(T,record_M_total_nodes)
plt.ylabel("Total mass of soil property on grid")
plt.xlabel("Time (y)")
plt.show()

plt.plot(T,record_M_total_links)
plt.ylabel("Total mass of soil property on links")
plt.xlabel("Time (y)")
plt.show()

plt.plot(T,record_M_out)
plt.ylabel("Total mass of soil property exiting bottom link")
plt.xlabel("Time (y)")
plt.show()

plt.plot(T,record_Soil_flux)
plt.ylabel("Soil flux exiting bottom link")
plt.xlabel("Time (y)")
plt.show()

plt.plot(T,record_H_total_nodes)
plt.ylabel("Total depth of soil on grid")
plt.xlabel("Time (y)")
plt.show()

mass_conserved_fraction = np.min(record_M_total_nodes)/record_M_total_nodes[0]
print("Fraction of mass conserved:   " + str(round(mass_conserved_fraction,3)))

# %% Plot final hillslope

# imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
# plt.show()

