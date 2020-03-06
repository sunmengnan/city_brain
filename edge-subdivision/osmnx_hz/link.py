import geojson
from pprint import pprint
import osmnx
import os
import hashids
from hashids import Hashids
from intersection import osm_node_way_df_from_response_jsons

hashids = Hashids(salt='this is my salt 1')
BASE_DIR = os.path.abspath(os.path.join(__file__, "../.."))
PATH = os.path.join(BASE_DIR, 'data/binjiang1.osm')

def process_edge_row(row):
    osmid = row['osmid']
    if not isinstance(osmid,list):
        row['osmid']=[osmid,]
    id1 = gen_osm_id(*row['osmid'],row['u'],row['v'])
    row['id1'] = id1
    row['geometry'] = geojson.dumps(row['geometry'])
    return row

def gen_osm_id(*args):
    # need add space
    hashid = hashids.encode(*args)
    return hashid

osm_graph = osmnx.graph_from_file(PATH,simplify=False)
overpass_json = osmnx.overpass_json_from_file(PATH)
osm_nodes,osm_ways = osm_node_way_df_from_response_jsons([overpass_json,])
simple_graph = osmnx.simplify.simplify_graph(osm_graph)
node_df,edge_df = osmnx.save_load.graph_to_gdfs(simple_graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)

pprint(edge_df)

link_df = edge_df.apply(process_edge_row,axis=1)
pprint(link_df)