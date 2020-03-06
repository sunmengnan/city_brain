import pymongo
import re
import geopy.distance


def get_distance(origin_point, new_point):
    dist = geopy.distance.distance(origin_point, new_point).km
    return dist


def collinear(point_a, point_b, point_c):
    """
    # Basically, it is to check that the slopes between point a and point b and point a and point c match.
    Slope is change in y divided by change in x
    :param point_a: tuple or list
    :param point_b: tuple or list
    :param point_c: tuple or list
    :return: bool
    """
    return (point_a[1] - point_b[1]) * (point_a[0] - point_c[0]) == \
           (point_a[1] - point_c[1]) * (point_a[0] - point_b[0])


myclient = pymongo.MongoClient("mongodb://192.168.120.127:27017")
mydb = myclient["tmap1"]
mycol = mydb["link"]

max_distance, count_limit, count_colinear, count_edge, count_edge_tobecut = 0, 0, 0, 0, 0

for x in mycol.find():
    count_edge += 1
    geometry = x["geometry"]
    coodinates = re.findall('\d{2,3}\.\d+', geometry)
    coodinates_array = []
    for i in range(0, len(coodinates), 2):
        coodinates_array.append((float(coodinates[i + 1]), float(coodinates[i])))

    # calculte the furthest distance of one edge
    fur_distance = get_distance(coodinates_array[0], coodinates_array[-1])

    for j in range(len(coodinates_array) - 1):
        distance = get_distance(coodinates_array[j], coodinates_array[j + 1])
        max_distance = max(max_distance, distance)
        if get_distance(coodinates_array[j], coodinates_array[j + 1]) > 3:
            print("exceed the limit")
            count_limit += 1

    if fur_distance > 3 and len(coodinates_array) > 2:
        count_edge_tobecut += 1
        for m in range(len(coodinates_array) - 2):
            if collinear(coodinates_array[m], coodinates_array[m + 1], coodinates_array[m + 2]):
                print('the {}th points, {},{},{} are collinear'.format(m,
                                                                       coodinates_array[m],
                                                                       coodinates_array[m + 1],
                                                                       coodinates_array[m + 2]))
                print('the distance are {},{}'.format(get_distance(coodinates_array[m], coodinates_array[m + 1]),
                                                      get_distance(coodinates_array[m + 1], coodinates_array[m + 2])))
                count_colinear += 1

print('The maximum length of a link is {}, the count limit is {}, the number of colinear points are {}, '
      'the total number of edges is {}, the total number of edges to be cut is {}.'.format(max_distance,
                                                                                           count_limit,
                                                                                           count_colinear,
                                                                                           count_edge,
                                                                                           count_edge_tobecut))
