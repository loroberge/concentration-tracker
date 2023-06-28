# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:35:27 2023

@author: LaurentRoberge
"""

import numpy as np
from collections import defaultdict

from landlab import Component
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
        
        # "LS__erosion_bedrock": "aaa",
        # "LS__erosion_sediment": "aaa",

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
        
        self._ls = bedrocklandslider_instance

        # use setters for C_init, C_br, P, and D defined below
        self.C_init = concentration_initial
        self.C_br = concentration_in_bedrock
        self.P = local_production_rate
        self.D = local_decay_rate
        
        # get reference to inputs
        self._soil__depth = self._grid.at_node["soil__depth"]
        self._soil__depth_old = self._soil__depth.copy()
        self._br_elev = self._grid.at_node["bedrock__elevation"]
        self._br_elev_old = self._br_elev.copy()

        # Define variables used for internal calculations
        self._cell_area = self._grid.dx*self._grid.dy
       
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


    def _concentration_eroded(self, dt):
        """Calculate concentration times volume for material eroded by every 
        landslide within the time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
                
        # Define values generated by BedrockLandslider
        F_f = self._ls._fraction_fines_LS
        phi = self._ls._phi
        self._critical_nodes = self._ls.critical_landslide_node_ids
        eroded_nodes = self._ls._eroded_node_ids
        
        # Calculate concentration*volume for bedrock and sediment layers
        V_br_eroded = (self._br_elev_old - self._br_elev) * self._cell_area
        self._CV_br_eroded = self.C_br * (1 - F_f) * V_br_eroded
        self._V_sed_eroded = (self._soil__depth_old - self._soil__depth) * self._cell_area
        self._CV_sed_eroded = self._C_old * (1 - phi) * self._V_sed_eroded
        
        # Empty dicts for lists of sliding nodes, CVsed, CVbr for each critical node
        CV_br_eroded_slide = defaultdict(list)
        CV_sed_eroded_slide = defaultdict(list)
        V_eroded = defaultdict(list)
        self._C_eroded = defaultdict(list)
        
        # Associate eroded nodes with their critical node ID
        for crit_node in self._critical_nodes:

            # Calculate eroded concentrations and volumes for each landslide
            CV_br_eroded_slide[crit_node] = np.sum(self._CV_br_eroded[eroded_nodes[crit_node]])
            CV_sed_eroded_slide[crit_node] = np.sum(self._CV_sed_eroded[eroded_nodes[crit_node]])
            V_eroded[crit_node] = np.sum(V_br_eroded[eroded_nodes[crit_node]]+self._V_sed_eroded[eroded_nodes[crit_node]])
            
            # Calculate concentration per unit volume eroded to each critical node
            # (to be passed through node and sent to deposits)
            self._C_eroded[crit_node] = (
                (CV_br_eroded_slide[crit_node] + CV_sed_eroded_slide[crit_node])
                / V_eroded[crit_node]
                )
            
        return self._C_eroded
        
    def _concentration_deposited(self, dt):
        """Calculate concentration times volume for material deposited by every 
        landslide within the time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
              
        # Define values
        self._CV_sed_dep = np.zeros(np.shape(self._soil__depth))
        
        # Define values generated by BedrockLandslider
        self._deposit_nodes = self._ls._deposited_node_ids
        
        # Calculate volume deposited at each node on grid
        self._V_sed_dep = self._grid.at_node["landslide__deposition"] * self._cell_area

        # Calculate absolute concentration prior to landsliding
        self._CV_old = self._C_old * self._soil__depth_old * self._cell_area
        
        # Calculate deposited concentration*volume for each slide deposit
        for crit_node in self._critical_nodes:
            dep_nodes = self._deposit_nodes[crit_node]
            self._CV_sed_dep[dep_nodes] = self._C_eroded[crit_node] * self._V_sed_dep[dep_nodes]

            
    def concentration(self, dt):
        """Calculate change in concentration for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """
        
        # Get array of all nodes that underwent erosion and deposition
        self._erode_nodes_all = self._grid.core_nodes[self._grid.at_node["landslide__erosion"][self._grid.core_nodes]>0]
        self._deposit_nodes_all = self._grid.core_nodes[self._grid.at_node["landslide__deposition"][self._grid.core_nodes]>0]

        # Calculate new concentration values for eroded nodes
        self._concentration[self._erode_nodes_all] = (
            (self._CV_old[self._erode_nodes_all] - self._CV_sed_eroded[self._erode_nodes_all])
            / (self._soil__depth[self._erode_nodes_all] * self._cell_area)
            )
        
        # Calculate new concentration values for deposited nodes
        self._concentration[self._deposit_nodes_all] = (
            (self._CV_old[self._deposit_nodes_all] + self._CV_sed_dep[self._deposit_nodes_all])
            / (self._soil__depth[self._deposit_nodes_all] * self._cell_area)
            )
            ### CHECK THAT THIS ALLOWS FOR MULTIPLE LANDSLIDES DEPOSITING 
            ### ON TOP OF EACH OTHER DURING ONE TIMESTEP
            
        # Replace nan values with zeros where that soil depth is zero
        idx_nans = np.isnan(self._concentration)
        idx_zero_soil = self._soil__depth==0
        self._concentration[idx_nans & idx_zero_soil] = 0
        
        # This deals with nans caused by a divide by zero in the previous 2 equations
        
        # Check that deposited and eroded C and V are balanced
        from numpy import testing
        err_msg = "Error in mass balance between C and V eroded vs deposited."
        testing.assert_almost_equal(
            np.sum(self._CV_sed_dep[self._deposit_nodes_all]),
            np.sum(self._CV_br_eroded[self._erode_nodes_all]
                   + self._CV_sed_eroded[self._erode_nodes_all]),
            decimal=10,
            err_msg=err_msg
            )
        
        # Update old soil depth and concentrations to new values
        self._soil__depth_old = self._soil__depth.copy()
        self._br_elev_old = self._br_elev.copy()
        self._C_old = self._concentration.copy()

    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        
        C_eroded = self._concentration_eroded(dt)
        self._concentration_deposited(dt)
        self.concentration(dt)
        
        return C_eroded