import pandas as pd
from pandas.core.common import flatten
import importlib
import pandas as pd
import osmnx
import numpy as np
import matplotlib.pyplot as plt
import geojson
import networkx
from pprint import pprint
from shapely.geometry import LineString, Polygon, Point
import shapely
import pymongo

districts = [
    {'city': '杭州市', 'district': '上城区'},
    {'city': '杭州市', 'district': '下城区'},
    {'city': '杭州市', 'district': '拱墅区'},
    {'city': '杭州市', 'district': '西湖区'},
    {'city': '杭州市', 'district': '江干区'},
    {'city': '杭州市', 'district': '滨江区'},
]

myclient = pymongo.MongoClient("mongodb://192.168.120.127:27017")
mydb = myclient["tmap2"]
mylink = mydb["link1"]
myway = mydb["osm_edge"]
mylink_update = mydb["link2"]


def get_district_gdf(districts):
    district_docs = []
    for district in districts:
        doc = mydb.district.find_one(district)
        district_docs.append(doc)
    district_gdf = pd.DataFrame(district_docs)
    return district_gdf


district_gdf = get_district_gdf(districts)
