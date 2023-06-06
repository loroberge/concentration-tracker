# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:41:20 2023

@author: LaurentRoberge
"""

import numpy as np

from landlab import Component, LinkStatus
from landlab.grid.mappers import map_value_at_max_node_to_link


class ConcentrationTracker(Component):

    """This component tracks the concentration of any user-defined property of
    sediment using a mass balance approach in which the concentration :math:`C`
    is calculated as:

    .. math::

        ∂CH/∂t = [-(∂q_x C_x)/∂x - (∂q_y C_y)/∂y] + PH + DH
        
    where :math:`H` is sediment depth, :math:`q_x` and :math:`q_y` are sediment
    fluxed in the x and y directions, :math:`P` is the local production rate,
    :math:`D` is the local decay rate.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import ExponentialWeatherer
    >>> from landlab.components import DepthDependentDiffuser
    >>> from landlab.components import ConcentrationTracker
    >>> ... SIMPLE EXAMPLE
    
    Now for a 1-D normal fault scarp undergoing hillslope diffusion:

    >>> ... EXAMPLE  
    
    Now, a 2-D landscape with stream channels and hillslopes.

    >>> ... EXAMPLE  

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    CITATION

    """

    _name = "ConcentrationTracker"

    _unit_agnostic = True

    _cite_as = """
    CITATION
    """

    _info = {
        "bedrock__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "elevation of the bedrock surface",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of soil or weathered bedrock",
        },
        "soil__flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m^2/yr",
            "mapping": "link",
            "doc": "flux of soil in direction of link",
        },
        "soil_production__rate": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/yr",
            "mapping": "node",
            "doc": "rate of soil production at nodes",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__slope": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "link",
            "doc": "gradient of the ground surface",
        },
    }

    def __init__(self, grid, local_production_rate=0, local_decay_rate=0):
        """
        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object
        local_production_rate: float
            Rate of local production, kg**3/yr
        local_decay_rate: float
            Rate of local decay, kg**3/yr
        """
        super().__init__(grid)
        # Store grid and parameters

        self._P = local_production_rate
        self._D = local_decay_rate
        
        # NOT SURE WHERE TO PUT THIS.... OUTPUT?
        self._soil__depth_old = np.zeros(grid.number_of_nodes)
        self._soil__depth_old += self._grid.at_node["soil__depth"]

        # get reference to inputs
        self._elev = self._grid.at_node["topographic__elevation"]
        self._soil__depth = self._grid.at_node["soil__depth"]
        self._flux = self._grid.at_node["sediment__flux"]
        self._flux = self._grid.at_link["soil__flux"]

        # create outputs if necessary and get reference.
        self.initialize_output_fields()
        self._concentration = self._grid.at_node["sed_property__concentration"]
        # self._slope = self._grid.at_link["topographic__slope"]
        # self._bedrock = self._grid.at_node["bedrock__elevation"]

    def concentration(self, dt):
        """Calculate change in concentration for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
                
        # Define soil depth at current time
        C_old = self._concentration
        
        # Map concentration from nodes to links (following soil flux direction)
        self._grid.at_link['C'] = map_value_at_max_node_to_link(
            self._grid,'topographic__elevation','Sm__concentration'
            )
        
        # Calculate QC at links (sediment flux times concentration)
        self._grid.at_link['QC'] = (self._grid.at_link['soil__flux'] *
                                    self._grid.at_link['C']
                                    )
        
        # Replace nan values with zeros (DOUBLE CHECK IF THIS IS NECESSARY)
        self._grid.at_link['QC'][np.isnan(self._grid.at_link['QC'])] = 0
        
        # Calculate flux concentration divergence
        dQCdx = self._grid.calc_flux_div_at_node(self._grid.at_link['QC'])
        
        # Calculate other components of mass balance equation
        C_local = C_old * (self._soil__depth_old/self._soil__depth)
        Production = (dt*self._P/2) * (self._soil__depth_old/self._soil__depth + 1)
        Decay = (dt*self._D/2) * (self._soil__depth_old/self._soil__depth + 1)
        
        # Calculate concentration
        self._concentration = (C_local + (dt/self._soil__depth) * (- dQCdx)
                               + Production - Decay
                               )
        
        # Update old soil depth to new value
        self._soil__depth_old = self._soil__depth


    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        self.concentration(dt)
