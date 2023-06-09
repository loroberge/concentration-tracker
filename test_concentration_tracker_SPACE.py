# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:25:27 2023

@author: LaurentRoberge
"""

import pytest
import numpy as np
from landlab import FieldError, RasterModelGrid
from landlab.components import ConcentrationTrackerForSpace

# %% Test input field errors

def test_input_fluxes_from_space():
    """
    ConcentrationTrackerForSpace should throw an error when output fields from
    a Space component do not exist (sediment__influx and sediment__outflux)
    """
    # Make a raster model grid 
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")

    fields_to_add = ["sediment__influx",
                     "sediment__outflux"]
        
    # Instantiate the component
    for field in fields_to_add:
        with pytest.raises(FieldError):
            _ = ConcentrationTrackerForSpace(mg)
        _ = mg.add_zeros(field, at="node")
        
def test_input_fields_soil():
    """
    ConcentrationTrackerForSpace should throw an error when input fields are
    not provided (soil__depth, soil_production__rate, topographic__elevation)
    """
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')

    fields_to_add = ["soil__depth",
                     "topographic__elevation"]
        
    # Instantiate the component
    for field in fields_to_add:
        with pytest.raises(FieldError):
            _ = ConcentrationTrackerForSpace(mg)
        _ = mg.add_zeros(field, at="node")

# %% Test field instantiation

def test_field_instantiation():
    """
    ConcentrationTrackerForSpace should instantiate the following fields
    when they do not already exist ('bedrock_property__concentration', 
    'sediment_property__concentration', 'sediment_property_decay__rate', 
    'sediment_property_production__rate')
    """
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")

    _ = ConcentrationTrackerForSpace(mg)
    
    node_fields = ['bedrock_property__concentration',
                   'sediment_property__concentration',
                   'sediment_property_decay__rate',
                   'sediment_property_production__rate'
                   ]
    
    for node_field in node_fields:
        assert node_field in mg.at_node
        
        
#%% Test different user input options

# Test that default input produces correct fields with no pre-existing fields
def test_fields_for_default_input():
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")
    
    _ = ConcentrationTrackerForSpace(mg)
    
    node_fields = [mg.at_node["sediment_property__concentration"],
                   mg.at_node["bedrock_property__concentration"],
                   mg.at_node["sediment_property_production__rate"],
                   mg.at_node["sediment_property_decay__rate"]
                   ]
    
    node_check = np.array([0.,  0.,  0.,
                           0.,  0.,  0.,
                           0.,  0.,  0.])
    
    for node_field in node_fields:
        np.testing.assert_equal(node_field, node_check)


# Test that default input uses correct fields with pre-existing fields
def test_fields_for_default_input_with_preexisting_fields():
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")
    
    _ = mg.add_ones('sediment_property__concentration', at='node')
    _ = mg.add_ones('bedrock_property__concentration', at='node')
    _ = mg.add_ones('sediment_property_production__rate', at='node')
    _ = mg.add_ones('sediment_property_decay__rate', at='node')
    
    _ = ConcentrationTrackerForSpace(mg)
    
    node_fields = [mg.at_node["sediment_property__concentration"],
                   mg.at_node["bedrock_property__concentration"],
                   mg.at_node["sediment_property_production__rate"],
                   mg.at_node["sediment_property_decay__rate"]
                   ]
    
    node_check = np.array([1.,  1.,  1.,
                           1.,  1.,  1.,
                           1.,  1.,  1.])
    
    for node_field in node_fields:
        np.testing.assert_equal(node_field, node_check)
    

# Test that user input of single values produces the correct fields
def test_fields_for_user_value_input():
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")
        
    _ = ConcentrationTrackerForSpace(mg,    
                                         concentration_initial=1,
                                         concentration_in_bedrock=1,
                                         local_production_rate=1,
                                         local_decay_rate=1,)
    
    node_fields = [mg.at_node["sediment_property__concentration"],
                   mg.at_node["bedrock_property__concentration"],
                   mg.at_node["sediment_property_production__rate"],
                   mg.at_node["sediment_property_decay__rate"]
                   ]
    
    node_check = np.array([1.,  1.,  1.,
                           1.,  1.,  1.,
                           1.,  1.,  1.])
    
    for node_field in node_fields:
        np.testing.assert_equal(node_field, node_check)
        
        
# Test that user input of arrays produces the correct fields
def test_fields_for_user_array_input():
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")
    
    c_sed = np.array([1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.])
    c_br = np.array([1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.])
    p = np.array([1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.])
    d = np.array([1.,  1.,  1.,  1.,  1.,  1., 1.,  1.,  1.])
    
    _ = ConcentrationTrackerForSpace(mg,    
                                         concentration_initial=c_sed,
                                         concentration_in_bedrock=c_br,
                                         local_production_rate=p,
                                         local_decay_rate=d,)
    
    node_fields = [mg.at_node["sediment_property__concentration"],
                   mg.at_node["bedrock_property__concentration"],
                   mg.at_node["sediment_property_production__rate"],
                   mg.at_node["sediment_property_decay__rate"]
                   ]

    node_check = np.array([1.,  1.,  1.,
                           1.,  1.,  1.,
                           1.,  1.,  1.])
    
    for node_field in node_fields:
        np.testing.assert_equal(node_field, node_check)
        
        
# Test that user input of grid fields produces the correct fields
def test_fields_for_user_field_input():
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")
    
    c_sed = mg.add_ones('sediment_property__concentration', at='node')
    c_br = mg.add_ones('bedrock_property__concentration', at='node')
    p = mg.add_ones('sediment_property_production__rate', at='node')
    d = mg.add_ones('sediment_property_decay__rate', at='node')
    
    _ = ConcentrationTrackerForSpace(mg,    
                                         concentration_initial=c_sed,
                                         concentration_in_bedrock=c_br,
                                         local_production_rate=p,
                                         local_decay_rate=d,)
    
    node_fields = [mg.at_node["sediment_property__concentration"],
                   mg.at_node["bedrock_property__concentration"],
                   mg.at_node["sediment_property_production__rate"],
                   mg.at_node["sediment_property_decay__rate"]
                   ]
    
    node_check = np.array([1.,  1.,  1.,
                           1.,  1.,  1.,
                           1.,  1.,  1.])
    
    for node_field in node_fields:
        np.testing.assert_equal(node_field, node_check)


# Test that physically impossible inputs raise correct errors
def test_properties_concentrations():
    """
    ConcentrationTrackerForSpace should throw an error when input 
    concentration values are negative.
    """
    mg = RasterModelGrid((3, 3))
    _ = mg.add_zeros('sediment__influx', at='node')
    _ = mg.add_zeros('sediment__outflux', at='node')
    _ = mg.add_zeros("soil__depth", at="node")
    _ = mg.add_zeros("topographic__elevation", at="node")

    # Instantiate the component
    with pytest.raises(ValueError):
        _ = ConcentrationTrackerForSpace(mg, concentration_initial=-1)
    # Instantiate the component
    with pytest.raises(ValueError):
        _ = ConcentrationTrackerForSpace(mg, concentration_in_bedrock=-1)

# %% Test against analytical solutions
# PLACEHOLDER: Test results against 1-D analytical solution (for Space)
# (I think this is covered by the docstring tests, so I haven't added it here)

# PLACEHOLDER: Test results against 1-D analytical solution (for SpaceLargeScaleEroder)
# (I think this is covered by the docstring tests, so I haven't added it here)
