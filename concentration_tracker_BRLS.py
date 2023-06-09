# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:35:27 2023

@author: LaurentRoberge
"""

import numpy as np

from landlab import Component, LinkStatus
from landlab.grid.mappers import map_value_at_max_node_to_link
from landlab.utils.return_array import return_array_at_node


class ConcentrationTrackerBRLS(Component):

    """This component tracks the concentration ......... blah blah.......
    
    

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import ExponentialWeatherer
    >>> from landlab.components import DepthDependentDiffuser
    >>> from landlab.components import ConcentrationTracker
    >>> ... SIMPLE EXAMPLE (1-node landslide)
    
    Now for a 1-D landslide:

    >>> ... EXAMPLE  
    
    Now, a 2-D landslide:

    >>> ... EXAMPLE  

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    CITATION

    """

    _name = "ConcentrationTrackerBRLS"

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
        
        "LS__erosion_bedrock": "aaa",
        "LS__erosion_sediment": "aaa",

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
        """
        
        super().__init__(grid)
        # Store grid and parameters
           
        # use setters for C_init, C_br, P, and D defined below
        self.C_init = concentration_initial
        self.C_br = concentration_in_bedrock
        
        # get reference to inputs
        self._soil__depth = self._grid.at_node["soil__depth"]
        
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

    @C_init.setter
    def C_init(self, new_val):
        self._C_init = return_array_at_node(self._grid, new_val)
        
    @C_br.setter
    def C_br(self, new_val):
        self._C_br = return_array_at_node(self._grid, new_val)


    def concentration(self, dt):
        """Calculate change in concentration for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
                
MATH HERE


    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        self.concentration(dt)
