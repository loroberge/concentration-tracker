# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:21:47 2023

@author: LaurentRoberge
"""


# FOR d4 OR hex grid FLOW ROUTING:
    # Map concentration from nodes to links (following water flux direction)
    mg.at_link['C'] = map_value_at_max_node_to_link(mg,
                                                    'drainage_area',
                                                    'Sm__concentration'
                                                    )
    
    
# FOR d8, dinf FLOW ROUTING:
    # Map concentration from nodes to links (following water flux direction)
    mg.at_link['C'] = map_value_at_max_node_to_link(mg,
                                                    'drainage_area',
                                                    'Sm__concentration'
                                                    )
        "flow__receiver_node":
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)"
            
        "flow__receiver_proportions":
            "mapping": "node"
            "doc": "Node array of proportion of flow sent to each receiver."
    
    
        "flow__link_to_receiver_node":
            "intent": "out",
            "mapping": "node",
            "doc": "ID of link downstream of each node, which carries the discharge"
            
        "drainage_area":
            "intent": "out",
            "mapping": "node",
            "doc": "Upstream accumulated surface area contributing to the node's discharge"
            
        "topographic__steepest_slope":
            "mapping": "node",
            "doc": "The steepest *downhill* slope"
            
        "squared_length_adjacent":
            "mapping": "node"
            "doc":
                "Length to adjacent nodes, squared (calcualted in advance to "
                "save time during calculation"
    
# FOR hillslope FLOW ROUTING:
    # Map concentration from nodes to links (following water flux direction)
    mg.at_link['C'] = map_value_at_max_node_to_link(mg,
                                                    'hill_drainage_area',
                                                    'Sm__concentration'
                                                    )

        "hill_drainage_area":
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver."
            
        "hill_flow__receiver_node":
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)"

        "hill_topographic__steepest_slope":
            "mapping": "node",
            "doc": "The steepest *downhill* slope"
            
        "hill_flow__receiver_proportions":
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver."
