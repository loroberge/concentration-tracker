# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:41:20 2023

@author: LaurentRoberge
"""

import numpy as np

from landlab import Component, LinkStatus
from landlab.grid.mappers import map_value_at_max_node_to_link
from landlab.utils.return_array import return_array_at_node


class ConcentrationTracker(Component):

    """This component tracks the concentration of any user-defined property of
    sediment using a mass balance approach in which the concentration :math:`C`
    is calculated as:

    .. math::

        ∂CH/∂t = [-(∂q_x C_x)/∂x - (∂q_y C_y)/∂y] + C_br*H_brw + PH + DH
        
    where :math:`H` is sediment depth, :math:`q_x` and :math:`q_y` are sediment
    fluxed in the x and y directions, :math:`C_br` is concentration in parent 
    bedrock, :math:`H_brw` is the height of bedrock weathered into soil, 
    :math:`P` is the local production rate, :math:`D` is the local decay rate.

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
        "soil__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of soil or weathered bedrock",
        },
        "soil__flux": {
            "dtype": float,
            "intent": "in",
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
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "sed_property__concentration": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg/m^3",
            "mapping": "node",
            "doc": "Mass concentration of property per volume of sediment",
        }
    }

    def __init__(self, 
                 grid, 
                 concentration_initial=0, 
                 concentration_in_bedrock=0, 
                 local_production_rate=0, 
                 local_decay_rate=0
                 ):
        """
        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object
        concentration_initial: positive float, array, or field name (optional)
            Initial concentration in soil/sediment, kg/m^3
        concentration_in_bedrock: positive float, array, or field name (optional)
            Concentration in bedrock, kg/m^3
        local_production_rate: float, array, or field name (optional)
            Rate of local production, kg/m^3/yr
        local_decay_rate: float, array, or field name (optional)
            Rate of local decay, kg/m^3/yr
        """
        
        super().__init__(grid)
        # Store grid and parameters
           
        # use setters for C_init, C_br, P, and D defined below
        self.C_init = concentration_initial
        self.C_br = concentration_in_bedrock
        self.P = local_production_rate
        self.D = local_decay_rate
        
        # get reference to inputs
        self._soil__depth = self._grid.at_node["soil__depth"]
        self._soil__depth_old = self._soil__depth.copy()
        self._soil_prod_rate = self._grid.at_node["soil_production__rate"]
        self._flux = self._grid.at_link["soil__flux"]
        
        # create outputs if necessary and get reference.
        self.initialize_output_fields()
        
        # Define concentration field
        self._concentration = grid.at_node["sed_property__concentration"]
        
        # Sediment property concentration field (at links, to calculate dQCdx)
        if "C" in grid.at_link:
            self._C_links = grid.at_link["C"]
        else:
            self._C_links = grid.add_zeros("C", at="link", dtype=float)
        
        # Sediment property mass field (at links, to calculate dQCdx)
        if "QC" in grid.at_link:
            self._QC_links = grid.at_link["QC"]
        else:
            self._QC_links = grid.add_zeros("QC", at="link", dtype=float)
        
        # Check that concentration values are within physical limits
        if isinstance(concentration_initial, np.ndarray):
            if concentration_initial.any() < 0:
                raise ValueError("Concentration cannot be negative.")
        else:
            if concentration_initial < 0:
                raise ValueError("Concentration cannot be negative.")

        if isinstance(concentration_in_bedrock, np.ndarray):
            if concentration_in_bedrock.any() < 0:
                raise ValueError("Concentration in bedrock cannot be negative.")
        else:
            if concentration_in_bedrock < 0:
                raise ValueError("Concentration in bedrock cannot be negative.")
        
    @property
    def C_init(self):
        """Initial concentration in soil/sediment (kg/m^3)."""
        return self._C_init
    
    @property
    def C_br(self):
        """Concentration in bedrock (kg/m^3)."""
        return self._C_br
    
    @property
    def P(self):
        """Rate of local production (kg/m^3/yr)."""
        return self._P
    
    @property
    def D(self):
        """Rate of local decay (kg/m^3/yr)."""
        return self._D

    @C_init.setter
    def C_init(self, new_val):
        self._C_init = return_array_at_node(self._grid, new_val)
        
    @C_br.setter
    def C_br(self, new_val):
        self._C_br = return_array_at_node(self._grid, new_val)
        
    @P.setter
    def P(self, new_val):
        self._P = return_array_at_node(self._grid, new_val)
        
    @D.setter
    def D(self, new_val):
        self._D = return_array_at_node(self._grid, new_val)


    def concentration(self, dt):
        """Calculate change in concentration for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
                
        # Define concentration at previous timestep
        C_old = self._concentration.copy()
        
        # Map concentration from nodes to links (following soil flux direction)
        # Does this overwrite fixed-value/gradient links?
        self._grid.at_link['C'] = map_value_at_max_node_to_link(
            self._grid,'topographic__elevation','sed_property__concentration'
            )
        # Replace values with zero for all INACTIVE links
        self._grid.at_link['C'][self._grid.status_at_link == LinkStatus.INACTIVE] = 0.0
       
        # Replace nan values with zeros (DOUBLE CHECK IF THIS IS NECESSARY)
        #self._grid.at_link['C'][np.isnan(self._grid.at_link['C'])] = 0.0
        
        # Calculate QC at links (sediment flux times concentration)
        self._grid.at_link['QC'] = (self._grid.at_link['soil__flux'][:]*
                                    self._grid.at_link['C'][:]
                                    )
        # Replace nan values with zeros (DOUBLE CHECK IF THIS IS NECESSARY)
        #self._grid.at_link['QC'][np.isnan(self._grid.at_link['QC'])] = 0.0
        
        # Calculate flux concentration divergence
        dQCdx = self._grid.calc_flux_div_at_node(self._grid.at_link['QC'])
        
        # Calculate other components of mass balance equation
        C_local = C_old * (self._soil__depth_old/self._soil__depth)
        C_from_weathering = self._C_br * (self._soil_prod_rate * dt)/self._soil__depth
        Production = (dt*self._P/2) * (self._soil__depth_old/self._soil__depth + 1)
        Decay = (dt*self._D/2) * (self._soil__depth_old/self._soil__depth + 1)
        
        # Calculate concentration
        self._concentration[:] = (C_local 
                                  + C_from_weathering 
                                  + (dt/self._soil__depth) * (- dQCdx)
                                  + Production 
                                  - Decay
                                  )
                
        # Update old soil depth to new value
        self._soil__depth_old = self._soil__depth.copy()


    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        self.concentration(dt)
