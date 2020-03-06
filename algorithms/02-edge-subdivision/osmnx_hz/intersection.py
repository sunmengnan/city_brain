import osmnx
import os
import pandas as pd
import pprint


def osm_node_way_df_from_response_jsons(response_jsons):
    nodes = {}
    paths = {}
    for osm_data in response_jsons:
        nodes_temp, paths_temp = osmnx.parse_osm_nodes_paths(osm_data)
        for key, value in nodes_temp.items():
            nodes[key] = value
        for key, value in paths_temp.items():
            paths[key] = value
    nodes_df = pd.DataFrame.from_dict(nodes, orient='index')
    ways_df = pd.DataFrame.from_dict(paths, orient='index')
    return nodes_df, ways_df


BASE_DIR = os.path.abspath(os.path.join(__file__, "../.."))
PATH = os.path.join(BASE_DIR, 'data/binjiang1.osm')

osm_graph = osmnx.graph_from_file(PATH, simplify=False)
overpass_json = osmnx.overpass_json_from_file(PATH)
osm_nodes, osm_ways = osm_node_way_df_from_response_jsons([overpass_json, ])
simple_graph = osmnx.simplify.simplify_graph(osm_graph)
undirect_simple_graph = osmnx.save_load.get_undirected(simple_graph)
undirect_simple_nodes, undirect_simple_edges = osmnx.save_load.graph_to_gdfs \
    (undirect_simple_graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)

pprint.pprint(osm_nodes)
pprint.pprint(osm_ways)
