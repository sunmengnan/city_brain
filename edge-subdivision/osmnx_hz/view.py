import pymongo
import shapely
import folium
import copy
import pandas as pd
import geopandas as gpd
import geojson


def point_change_xy(point):
    new_point = copy.deepcopy(point)
    new_point.coordinates = point.coordinates[::-1]
    return new_point


myclient = pymongo.MongoClient("mongodb://192.168.120.127:27017")
mydb = myclient["tmap1"]
mycol = mydb["link"]

data = pd.DataFrame(list(mydb.intersection.find()))
data['geometry'] = data['centroid'].apply(geojson.loads).apply(shapely.geometry.asShape)
gpd_df = gpd.GeoDataFrame(data, geometry='geometry')
gpd_df.drop(columns=['_id'])

gpd_df1 = gpd_df.copy()

gpd_df1.crs = {'init': 'epsg:4326'}
print(gpd_df1.__geo_interface__)

style_function = lambda x: {'fillColor': '#0000ff', }

m = folium.Map([30.186518, 120.176028], zoom_start=13)
p = geojson.Point([120.176028, 30.186518])
g_p = folium.GeoJson(p, style_function=style_function)

g_p.add_to(m)
m.save('map_binjiang.html')
