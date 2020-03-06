# divid by length from experience estimation
import pymongo
import json
import pandas as pd
import geopandas as gpd
import hashids
from hashids import Hashids
from shapely.geometry import LineString, Polygon, Point
import shapely
import osmnx
import geojson
import geopy.distance

myclient = pymongo.MongoClient("mongodb://192.168.120.127:27017")
mydb = myclient["tmap"]
mylink = mydb["link2"]
data = pd.DataFrame(list(mydb.link2.find()))
data = data.sort_values(by=['length'], ascending=False)
data['geometry'] = data['geometry_geojson'].apply(shapely.geometry.asShape)
data = data.drop(columns=['_id'])
gdf1 = gpd.GeoDataFrame(data)
gdf2 = gdf1.iloc[:300]

districts = [
    {'city': '杭州市', 'district': '上城区'},
    {'city': '杭州市', 'district': '下城区'},
    {'city': '杭州市', 'district': '拱墅区'},
    {'city': '杭州市', 'district': '西湖区'},
    {'city': '杭州市', 'district': '江干区'},
    {'city': '杭州市', 'district': '滨江区'},
]


def gen_osm_id(*args):
    hashid = hashids.encode(*args)
    return hashid


hashids = Hashids(salt='this is my salt 1')


# load node and point relation

def get_district_gdf(districts):
    district_docs = []
    for district in districts:
        doc = mydb.district.find_one(district)
        district_docs.append(doc)
    district_gdf = pd.DataFrame(district_docs)
    return district_gdf


def osm_node_way_df_from_response_jsons(response_jsons):
    nodes, paths = {}, {}
    for osm_data in response_jsons:
        nodes_temp, paths_temp = osmnx.parse_osm_nodes_paths(osm_data)
        for key, value in nodes_temp.items():
            nodes[key] = value
        for key, value in paths_temp.items():
            paths[key] = value
    nodes_df = pd.DataFrame.from_dict(nodes, orient='index')
    ways_df = pd.DataFrame.from_dict(paths, orient='index')
    return nodes_df, ways_df


district_gdf = get_district_gdf(districts)
district_gdf = gpd.GeoDataFrame(district_gdf)
district_gdf['polygon'] = district_gdf['geojson'].apply(shapely.geometry.shape)
district_gdf = district_gdf.set_geometry('polygon', crs='EPSG:4326')
polygon_unary = district_gdf.polygon.unary_union
osm_json = osmnx.osm_net_download(polygon=polygon_unary, network_type='drive')
osm_node_df, osm_way_df = osm_node_way_df_from_response_jsons(osm_json)


# divid link
def calc_node_geometry_geojson(noderow):
    node_geometry_geojson = geojson.Point((noderow['x'], noderow['y']))
    return node_geometry_geojson


def calc_way_geometry_geojson(nodes):
    def get_node_xy(node_id):
        node = osm_node_df.loc[node_id]
        return node['x'], node['y']

    way_geometry_geojson = geojson.LineString([get_node_xy(node_id) for node_id in nodes])
    return way_geometry_geojson


def get_distance(origin_point, new_point):
    dist = geopy.distance.distance(tuple(origin_point), tuple(new_point)).m
    return dist


def extra_nodes(row):
    nodes = row['_nodes'].iloc[0]

    def divid_nodes_array(nodes):
        node_init, index_init = nodes[0], 0
        nodes_array = [node_init]
        nodes_array_dict = {}
        for i in range(1, len(nodes)):
            point_init = row.iloc[0]['geometry_geojson']['coordinates'][index_init][::-1]
            point_end = row.iloc[0]['geometry_geojson']['coordinates'][i][::-1]
            if get_distance(point_init, point_end) > 350 and i != len(nodes):
                nodes_array.append(nodes[i])
                nodes_array_dict[nodes[i]] = point_end
                index_init = i
            elif get_distance(point_init, point_end) < 350 and i != len(nodes):
                pass
            elif i == len(nodes):
                nodes_array.append(nodes[i])
        return nodes_array, nodes_array_dict

    # the edge name should be single and nodes should not be closed

    nodes_to_bedivid, nodes_dict = divid_nodes_array(nodes)
    nodes_to_bedivid = nodes_to_bedivid[1:-1]
    edge_divided = pd.DataFrame(columns=['osmid', 'x', 'y'])
    if nodes_to_bedivid:
        for item in nodes_to_bedivid:
            new_row = pd.DataFrame({'osmid': item, 'x': nodes_dict[item][0], 'y': nodes_dict[item][1]},
                                   index=[int(item)])
            edge_divided = edge_divided.append(new_row)
        edge_divided['highway'], edge_divided['ref'] = None, None
        return edge_divided

    return None


def divid_edge_row(row):
    nodes = row['_nodes'].iloc[0]

    def divid_nodes_array(nodes):
        node_init, index_init = nodes[0], 0
        nodes_array = [node_init]
        for i in range(1, len(nodes)):
            point_init = row.iloc[0]['geometry_geojson']['coordinates'][index_init][::-1]
            point_end = row.iloc[0]['geometry_geojson']['coordinates'][i][::-1]
            if get_distance(point_init, point_end) > 300 and i != len(nodes):
                nodes_array.append(nodes[i])
                index_init = i
            elif get_distance(point_init, point_end) < 300 and i != len(nodes):
                pass
            elif i == len(nodes):
                nodes_array.append(nodes[i])
        return nodes_array

    # the edge name should be single and nodes should not be closed

    nodes_to_bedivid = divid_nodes_array(nodes)
    edge_divided = pd.DataFrame()
    if len(nodes_to_bedivid) > 2:
        for j in range(len(nodes_to_bedivid) - 1):
            link_df_rowcp = row.copy()
            link_df_rowcp['u'], link_df_rowcp['v'] = nodes_to_bedivid[j], nodes_to_bedivid[j + 1]
            link_df_rowcp['id1'] = gen_osm_id(link_df_rowcp['osmid'], link_df_rowcp['u'],
                                              link_df_rowcp['v'])
            index1, index2 = nodes.index(nodes_to_bedivid[j]), nodes.index(nodes_to_bedivid[j + 1])
            link_df_rowcp['_nodes'] = nodes[index1:(index2 + 1)]
            link_df_rowcp['geometry_json'] = calc_way_geometry_geojson(link_df_rowcp['_nodes'])
            link_df_rowcp['geometry'] = link_df_rowcp['geometry_geojson'].apply(shapely.geometry.asShape)
            node_front = link_df_rowcp['geometry_json']['coordinates'][0][::-1]
            node_end = link_df_rowcp['geometry_json']['coordinates'][-1][::-1]
            link_df_rowcp['length'] = get_distance(node_front, node_end)
            link_df_rowcp['id2'] = gen_osm_id(*link_df_rowcp['_nodes'])
            link_df_rowcp = pd.DataFrame(link_df_rowcp).T
            edge_divided = pd.concat([edge_divided, link_df_rowcp])
        return edge_divided

    return pd.DataFrame(row).T


processed = extra_nodes(gdf2.head(1))
print(processed)

if __name__ == '__main__':

    # create a document
    link_df_update = pd.DataFrame()

    for index, row in link_df.iterrows():
        row_divid = divid_edge_row(row)
        print('row{} is done...'.format(index))
        link_df_update = pd.concat([link_df_update, divid_edge_row(row)], ignore_index=True)

    link_df_update = gpd.GeoDataFrame(link_df_update)

    # insert
    mydb.link2.delete_many({})
    mydb.link2.insert_many(link_df_update.to_dict('records'))
