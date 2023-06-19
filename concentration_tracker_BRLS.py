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

        "LS_sediment__flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux originating from landslides \
                (volume per unit time of sediment entering each node)",
        },
        "landslide__erosion": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Total erosion caused by landsliding ",
        },
        "landslide__deposition": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Total deposition of derived sediment",
        },
        "landslide_sediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3",
            "mapping": "node",
            "doc": "Landslide derived sediment, as point sources on all the \
                critical nodes where landslides initiate, \
                before landslide runout is calculated ",
        },
        "sed_property__concentration": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg/m^3",
            "mapping": "node",
            "doc": "Mass concentration of property per unit volume of sediment",
        }
    }


    def __init__(self, 
                 grid, 
                 bedrocklandslider_instance,
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
        """
        
        super().__init__(grid)
        # Store grid and parameters
        
        self._br = bedrocklandslider_instance

        # use setters for C_init, C_br, P, and D defined below
        self.C_init = concentration_initial
        self.C_br = concentration_in_bedrock
        self.P = local_production_rate
        self.D = local_decay_rate
        
        # get reference to inputs
        self._soil__depth = self._grid.at_node["soil__depth"]
        self._soil__depth_old = self._soil__depth.copy()

        # Define variables used for internal calculations
        self._cell_area = self._grid.dx*self._grid.dy
        # self._C_sw = np.zeros(self._grid.number_of_nodes)
        # self._QsCsw_in = np.zeros(self._grid.number_of_nodes)
        # self._QsCsw_out = np.zeros(self._grid.number_of_nodes)
        # self._BED_ero_depo_term = np.zeros(self._grid.number_of_nodes)
        
        # create outputs if necessary and get reference.
        self.initialize_output_fields()
        
        # Define concentration field (if all zeros, then add C_init)
        if not self._grid.at_node["sed_property__concentration"].any():
            self._grid.at_node["sed_property__concentration"] += self.C_init
        self._concentration = self._grid.at_node["sed_property__concentration"]
        # NOTE: not sure if this is how to do it... I guess I need the above 
        # lines for C_br, P, and D as well...
        self._C_old = self._concentration.copy()

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
                
        # Define values generated by BedrockLandslider
        F_f = self._br._F_f
        phi = self._br._phi

        # Calculate absolute concentration prior to landsliding
        CV_old = self._concentration_old * self._soil__depth_old * self._cell_area

        # Get arrays of critical node IDs and respective eroded/deposited nodes
        critical_nodes = 1
        erode_nodes = {critical_node1 : [erode nodes 1],
                       critical_node2 : [erode nodes 2],
                       }
        deposit_nodes = {critical_node1 : [deposit_nodes 1],
                         critical_node2 : [deposit_nodes 2],
                         }
        
        # Calculate eroded concentrations and volumes for each landslide
        for node in critical_nodes:
            CV_br[node] = np.sum(Vbr[erode_nodes[node]] * Cbr[erode_nodes[node]])
            CV_sed[node] = np.sum(Vsed[erode_nodes[node]] * Csed[erode_nodes[node]])
            C_eroded[node] = (CV_br[critical_nodes] + CV_sed[critical_nodes]) \
                / V_total[critical_nodes]
        
        # Calculate new concentration vales
        for node in critical_nodes:
            CV_dep[deposit_nodes[node]] = Vdeposited[deposit_nodes[node]] * C_eroded[node]
            self._concentration[deposit_nodes[node]] = \
                (CV_old[deposit_nodes[node]] + CV_dep[deposit_nodes[node]]) \
                    / self._soil__depth[deposit_nodes[node]]
            ### CHECK THAT THIS ALLOWS FOR MULTIPLE LANDSLIDES DEPOSITING 
            ### ON TOP OF EACH OTHER DURING ONE TIMESTEP
        
        # Update old soil depth and concentrations to new values
        self._soil__depth_old = self._soil__depth.copy()
        self._concentration_old = self._concentration.copy()

    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        self.concentration(dt)
