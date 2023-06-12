# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:35:26 2023

@author: LaurentRoberge
"""

import numpy as np

from landlab import Component, LinkStatus
from landlab.grid.mappers import map_value_at_max_node_to_link
from landlab.utils.return_array import return_array_at_node


class ConcentrationTrackerSPACE(Component):

    """This component tracks the concentration ...........

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import ExponentialWeatherer
    >>> from landlab.components import DepthDependentDiffuser
    >>> from landlab.components import ConcentrationTracker
    >>> ... SIMPLE 1-D EXAMPLE
    
    Now for a 1-D stream channel:

    >>> ... EXAMPLE  
    
    Now, a 2-D landscape with stream channels.

    >>> ... EXAMPLE  

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    CITATION

    """

    _name = "ConcentrationTrackerSPACE"

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
        "sediment__influx": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m^3/yr",
            "mapping": "node",
            "doc": "flux of sediment into node",
        },
        "sediment__outflux": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m^3/yr",
            "mapping": "node",
            "doc": "flux of sediment out of node",
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
        self._Qs_in = self._grid.at_node["sediment__influx"]
        self._Qs_out = self._grid.at_link["sediment__outflux"]
        
        # create outputs if necessary and get reference.
        self.initialize_output_fields()
        
        # Define concentration field
        self._concentration = grid.at_node["sed_property__concentration"]
        
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


    # def concentration_bed(self, dt):
    #     """Calculate change in concentration in sediment on the bed for a 
    #     time period 'dt'.

    #     Parameters
    #     ----------

    #     dt: float (time)
    #         The imposed timestep.
    #     """


    def concentration_watercolumn_and_bed(self, dt):
        """Calculate change in concentration within sediment transported in
        the water column for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
        # COLLECT THE FOLLOWING FROM SPACE WHEN IT RUNS ONE STEP
        
        
        Qs_in = self._grid.at_node['sediment__influx']
        Qs_out = self._mg.at_node['sediment__outflux']
        E_s = sp._Es
        E_r = sp._Er
        D_sw = Qs_in - (Qs_out + E_s + E_r)

        # Define concentration at previous timestep
        C_s_old = self._concentration.copy()
        
        # Calculate mass balance terms that don't need downstream iteration
        WC_Qs_term = dx/Qs_in
        BED_C_local = C_s_old * (self._soil__depth_old/self._soil__depth)
        BED_Production = (dt*self._P/2) * (self._soil__depth_old/self._soil__depth + 1)
        BED_Decay = (dt*self._D/2) * (self._soil__depth_old/self._soil__depth + 1)
        
        # I don't think P & D should occur in fluxing sed, since it is already
        # happening in the sediment layer below, which is interacting with Q
        
        # WC_Production = (dt*self._P/2) * (self._soil__depth_old/self._soil__depth + 1)
        # WC_Decay = (dt*self._D/2) * (self._soil__depth_old/self._soil__depth + 1)
        

        for node_id in stack:
            WC_C_int_us[node_id] = (C_s[node_id]*E_s[node_id]
                                    + C_r[node_id]*E_r[node_id]
                                    - C_sw[node_id]*D_sw[node_id]
                                    )
            # Calculate concentration
            self._C_sw[node_id] = Qs_term * SUM
                                 # + Production
                                 # - Decay
                             
            BED_Ero_depo[node_id] = (self._C_sw[node_id] * D_sw[node_id]
                                     - C_s_old[node_id] * E_s[node_id])
                         
                         

            # Calculate concentration
            self._concentration[:] = (C_local 
                                      + (dt/self._soil__depth) * Ero_depo
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

        self.concentration_bed(dt)
        self.concentration_watercolumn(dt)
