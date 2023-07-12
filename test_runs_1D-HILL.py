# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:37:49 2023

@author: LaurentRoberge
"""

import time
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import (ExponentialWeatherer,
                                DepthDependentDiffuser,
                                PriorityFloodFlowRouter,
                                BedrockLandslider
                                )

from concentration_tracker_DDD import ConcentrationTrackerDDD
from concentration_tracker_BRLS import ConcentrationTrackerBRLS

# %% Filter warnings
warnings.filterwarnings('ignore')

# %% Define input parameters
nrows = 3
ncols = 50

dx = 10
dy = dx
dt = 5

total_t = 15000

C_initial = 0.75
C_br = 0

P = 0
D = 0

U = 0.001

# Parameters for DDD
soil_production_maximum_rate = 0.001
soil_production_decay_depth = 1

# Parameters for BedrockLandslider
tLS = 100
F_f_LS = 0
phi = 0
angle_int_frict = 0.3

# Calculated from user inputs
uplift_per_step = U*dt
total_t = total_t #+ dt
ndt = int(total_t // dt)
hill_bottom_node = ncols
hill_bottom_link = (ncols*2)-1
node_next_to_hill_bottom = ncols+1

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
#mg.at_node['soil__depth'] += 1.5 # mg.node_x/4 
mg.at_node['soil__depth'][ncols + int(3*ncols/4)-1] += 2 

# Bedrock elevation
_ = mg.add_zeros('bedrock__elevation', at='node', units= ['m','m'])
mg.at_node['bedrock__elevation'] += mg.node_x/50
mg.at_node['bedrock__elevation'][mg.node_x > dx*ncols/2] += 30

# Topographic elevation
_ = mg.add_zeros('topographic__elevation', at='node', units= ['m','m'])
mg.at_node['topographic__elevation'][:] += mg.at_node['bedrock__elevation']
mg.at_node['topographic__elevation'][:] += mg.at_node['soil__depth']

# Magnetic susceptibility concentration field
C = mg.add_zeros('sed_property__concentration', at='node', units= ['kg/m^3','kg/m^3'])
mg.at_node['sed_property__concentration'] += 0#C_initial
# mg.at_node['sed_property__concentration'][mg.node_x > ncols/2] += C_initial
mg.at_node['sed_property__concentration'][ncols + int(3*ncols/4)-1] += C_initial
# mg.at_node['sed_property__concentration'][ncols + int(3*ncols/4)] += C_initial
# mg.at_node['sed_property__concentration'][ncols + int(3*ncols/4)+1] += C_initial

core_ids = np.append(hill_bottom_node, mg.core_nodes)
n_core_nodes = len(mg.core_nodes)

# %% Instantiate model components
fr = PriorityFloodFlowRouter(mg,
                             accumulate_flow_hill=True,
                             separate_hill_flow=True,
                             update_hill_depressions=False,
                             update_hill_flow_instantaneous=True,
                             hill_flow_metric='Quinn',
                             )
fr.run_one_step()

ew = ExponentialWeatherer(mg,
                          soil_production_maximum_rate=soil_production_maximum_rate,
                          soil_production_decay_depth=soil_production_decay_depth
                          )

ddd = DepthDependentDiffuser(mg)

ls = BedrockLandslider(mg,
                       landslides_return_time=tLS,
                       fraction_fines_LS=F_f_LS,
                       phi=phi,
                       angle_int_frict=angle_int_frict,
                       verbose_landslides=True,
                       landslides_on_boundary_nodes=False,
                       critical_sliding_nodes=None,
                       output_landslide_node_ids=True,
                       )

ctDDD = ConcentrationTrackerDDD(mg,
                             concentration_initial=C,
                             concentration_in_bedrock=C_br,
                             local_production_rate=P,
                             local_decay_rate=D,
                             )

ctLS = ConcentrationTrackerBRLS(mg,
                                ls,
                                concentration_initial=C,
                                concentration_in_bedrock=C_br,
                                local_production_rate=P,
                                local_decay_rate=D,
                                )

#%% Create empty arrays to fill during loop

T = np.zeros(ndt)       # Time 

ymax = np.max(mg.at_node["topographic__elevation"][core_ids])+5

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

# imshow_grid(mg, "topographic__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "bedrock__elevation", cmap=cmap_terrain, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "soil__depth", cmap=cmap_soil, color_for_closed='pink')
# plt.show()
# imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink')
# plt.show()

plot_hill_profile()

# %% Model Run

# plot_hill_profile()

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
    
    # Run DepthDependentDiffuser
    fr.run_one_step()
    
    # Run BedrockLandslider
    ls.run_one_step(dt=dt)
        
    # Calculate concentration
    C_eroded = ctLS.run_one_step(dt=dt)
        
    plot_hill_profile()

    # if i*dt % 10 == 0:
        
    #     plot_hill_profile()
        
    #     # imshow_grid(mg, "sed_property__concentration", cmap=cmap_Sm, color_for_closed='pink')
    #     # plt.show()
    
    # New topo and concentration
    topo_new = mg.at_node['topographic__elevation'][mg.core_nodes].copy()    
    C_new = mg.at_node['sed_property__concentration'][mg.core_nodes].copy()    
    
    # Collect stats
    
    T[i] = elapsed_time

