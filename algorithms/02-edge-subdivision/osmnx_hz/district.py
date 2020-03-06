import pandas as pd
import osmnx
import numpy as np

fix = {'西湖区,杭州市,浙江省,中国': 2}
city_query = [
    '杭州市,浙江省,中国',
]
district_query = [
    '上城区,杭州市,浙江省,中国',
    '下城区,杭州市,浙江省,中国',
    '江干区,杭州市,浙江省,中国',
    '西湖区,杭州市,浙江省,中国',
    '拱墅区,杭州市,浙江省,中国',
    '滨江区,杭州市,浙江省,中国',
]


def query_str_to_dic(query_str):
    result = query_str.split(',')
    if len(result) == 3:
        result.insert(0, '')
    query_dic = {
        'district': result[0],
        'city': result[1],
        'province': result[2],
    }
    return query_dic


def process_query(q):
    query_dic = query_str_to_dic(q)
    limit = fix.get(q, 1)
    nominatim_response = osmnx.osm_polygon_download(q, limit=limit)
    response_json = nominatim_response[limit - 1]
    result_dic = {}
    result_dic.update(response_json)
    result_dic.update(query_dic)
    result_dic['q'] = q
    return result_dic


district_df = pd.DataFrame()
q_result_list = []
for q in district_query:
    q_result = process_query(q)
    q_result_list.append(q_result)
district_df = pd.DataFrame(q_result_list)
print(district_df)
