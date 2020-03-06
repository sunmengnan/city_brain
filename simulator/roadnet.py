try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import json


ROOT = ET.Element
SELECTED_INTERSECTION = []
VIRTUAL = ["3710259104", "3921474903", "5143576583", "3710240891"]
INTERSECTIONS = {}
ROADS = {}
LANELINKS = {}
LANE_NUMBER = {}
INTERNAL_LANE_SHAPE = {}


class intersection(object):
    def __init__(self):
        self.id = ""
        self.point = {}
        self.inRoads = []
        self.outRoads = []
        self.width = 0
        self.roadLinks = []
        self.trafficLight = {}
    def set_id(self, id):
        self.id = id
    def set_point(self, x, y):
        self.point['x'] = x
        self.point['y'] = y
    def insert_inroad(self, road):
        self.inRoads.append(road)
    def insert_outroad(self, road):
        self.outRoads.append(road)
    def insert_roadlink(self, roadLink):
        merge = False
        for i in range(len(self.roadLinks)):
            if self.roadLinks[i]["type"] == roadLink["type"] and self.roadLinks[i]["startRoad"] == roadLink["startRoad"] and self.roadLinks[i]["endRoad"] == roadLink["endRoad"]:
                self.roadLinks[i]["laneLinks"].append(roadLink["laneLinks"][0])
                merge = True
                break
        if not merge:
            self.roadLinks.append(roadLink)
    def set_default_traffic_light(self):
        self.trafficLight["roadLinkIndices"] = range(0, len(self.roadLinks))
        self.trafficLight["lightphases"] = []
        self.trafficLight["lightphases"].append({"availableRoadLinks": self.trafficLight["roadLinkIndices"], "time": 30})
    def set_lightphases(self, lightPhases):
        self.trafficLight["lightphases"] = lightPhases
    def get_inRoads(self):
        return self.inRoads
    def get_outRoads(self):
        return self.outRoads


class road(object):
    def __init__(self):
        self.id = ""
        self.points = []
        self.lanes = []
        self.startIntersection = ""
        self.endIntersection = ""
    def set_id(self, id):
        self.id = id
    def set_points(self, points):
        self.points = points
    def shift_road(self):
        class vector:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def __add__(self, other):
                return vector(self.x+other.x, self.y+other.y)
            def __sub__(self, other):
                return vector(self.x-other.x, self.y-other.y)
            def __mul__(self, constant):
                return vector(self.x*constant, self.y*constant)
            def normalize(self):
                l = (self.x ** 2 + self.y ** 2) ** 0.5
                return vector(self.x/l, self.y/l)
            def orthogonal_normalize(self):
                return vector(self.y, self.x*(-1)).normalize()
            def reverse(self):
                return vector(self.x*(-1), self.y*(-1))
            def get_x(self):
                return self.x
            def get_y(self):
                return self.y

        roadWidth = 0
        for lane in self.lanes:
            roadWidth += lane["width"]
        temp = []
        for i in range(len(self.points)):
            if i == 0:
                v1 = vector(self.points[i]['x'], self.points[i]['y'])
                v2 = vector(self.points[i+1]['x'], self.points[i+1]['y'])
                v3 = (v2-v1).orthogonal_normalize().reverse()
                v4 = v1 + v3 * (roadWidth/2)
                temp.append({'x': v4.get_x(), 'y': v4.get_y()})
            elif i == len(self.points) - 1:
                v1 = vector(self.points[i-1]['x'], self.points[i-1]['y'])
                v2 = vector(self.points[i]['x'], self.points[i]['y'])
                v3 = (v2-v1).orthogonal_normalize().reverse()
                v4 = v2 + v3 * (roadWidth/2)
                temp.append({'x': v4.get_x(), 'y': v4.get_y()})
            else:
                v1 = vector(self.points[i-1]['x'], self.points[i-1]['y'])
                v2 = vector(self.points[i]['x'], self.points[i]['y'])
                v3 = vector(self.points[i+1]['x'], self.points[i+1]['y'])
                v4 = (v3-v1).orthogonal_normalize().reverse()
                v5 = v2 + v4 * (roadWidth/2)
                temp.append({'x': v5.get_x(), 'y': v5.get_y()})
        self.points = temp
    def insert_lane(self, width, maxSpeed):
        self.lanes.append({"width": width, "maxSpeed": maxSpeed})
    def reverse_lanes(self):
        self.lanes.reverse()
    def set_startintersection(self, intersection):
        self.startIntersection = intersection
    def set_endintersection(self, intersection):
        self.endIntersection = intersection


class lanelink(object):
    def __init__(self):
        self.startRoad = ""
        self.endRoad = ""
        self.startLaneIndex = -1
        self.endLaneIndex = -1
        self.direction = ""
    def set_startroad(self, road):
        self.startRoad = road
    def set_endroad(self, road):
        self.endRoad = road
    def set_startlaneindex(self, lane):
        self.startLaneIndex = lane
    def set_endlaneindex(self, lane):
        self.endLaneIndex = lane
    def set_direction(self, direction):
        self.direction = direction
    def get_startroad(self):
        return self.startRoad
    def get_endroad(self):
        return self.endRoad
    def get_startlaneindex(self):
        return self.startLaneIndex
    def get_endlaneindex(self):
        return self.endLaneIndex
    def get_direction(self):
        return self.direction


# ===== Load manually selected intersections. =====
def load_selected_intersection(directory):
    global SELECTED_INTERSECTION
    for line in open(directory, 'r'):
        SELECTED_INTERSECTION.append(line.strip())


# ===== Load SUMO roadnet. =====
def load_sumo_roadnet(directory):
    global ROOT
    tree = ET.parse(directory)
    ROOT = tree.getroot()


# ===== Parse shape of a road or a lane. =====
def parse_shape(shape):
    points = []
    shapeSplit = shape.split(' ')
    for point in shapeSplit:
        pointSplit = point.split(',')
        points.append({'x': float(pointSplit[0]), 'y': float(pointSplit[1])})
    return points


# ===== Record the shape of internal edges and the number of lanes of non-internal edges. =====
def record_lane_info():
    global ROOT, LANE_NUMBER, INTERNAL_LANE_SHAPE
    for edge in ROOT.findall("edge"):
        if edge.get("function") == "internal":
            for lane in edge.findall("lane"):
                INTERNAL_LANE_SHAPE[lane.get("id")] = lane.get("shape")
        else:
            laneNumber = 0
            for lane in edge.findall("lane"):
                laneNumber += 1
            LANE_NUMBER[edge.get("id")] = laneNumber


# ===== Parse connection part of SUMO roadnet. =====
def parse_connection():
    global ROOT, LANELINKS
    for connection in ROOT.findall("connection"):
        if connection.get("via") != None:
            conncetionVia = connection.get("via")
            LANELINKS[conncetionVia] = lanelink()
            LANELINKS[conncetionVia].set_startroad(connection.get("from"))
            LANELINKS[conncetionVia].set_endroad(connection.get("to"))
            LANELINKS[conncetionVia].set_startlaneindex(int(connection.get("fromLane")))
            LANELINKS[conncetionVia].set_endlaneindex(int(connection.get("toLane")))
            LANELINKS[conncetionVia].set_direction(connection.get("dir"))


# ===== Parse junction part of SUMO roadnet (first step). =====
def parse_junction_1():
    global ROOT, SELECTED_INTERSECTION, INTERSECTIONS
    for junction in ROOT.findall("junction"):
        intersectionID = junction.get("id")
        if intersectionID in SELECTED_INTERSECTION:
            if junction.get("type") != "internal":
                INTERSECTIONS[intersectionID] = intersection()
                INTERSECTIONS[intersectionID].set_id(intersectionID)
                INTERSECTIONS[intersectionID].set_point(float(junction.get('x')), float(junction.get('y')))
            else:
                print ("[Warning] Junction " + intersectionID + " is of 'internal' type.")


# ===== Parse edge part of SUMO roadnet. =====
def parse_edge():
    global ROOT, SELECTED_INTERSECTION, INTERSECTIONS, ROADS
    for edge in ROOT.findall("edge"):
        if edge.get("function") != "internal" and edge.get("from") in SELECTED_INTERSECTION and edge.get("to") in SELECTED_INTERSECTION:
            roadID = edge.get("id")
            ROADS[roadID] = road()
            ROADS[roadID].set_id(roadID)
            if edge.get("shape") != None:
                edgeShape = edge.get("shape")
            else:
                for lane in edge.findall("lane"):
                    edgeShape = lane.get("shape")
                print ("[Warning] Shape of edge " + roadID + " is unknown. Use shape of first lane instead.")
            edgeShapePoints = parse_shape(edgeShape)
            ROADS[roadID].set_points(edgeShapePoints)
            for lane in edge.findall("lane"):
                if lane.get("width") != None:
                    ROADS[roadID].insert_lane(float(lane.get("width")), float(lane.get("speed")))
                else:
                    ROADS[roadID].insert_lane(4.0, float(lane.get("speed")))
                    print ("[Warning] Width of edge " + roadID + " is unknown. Use 4.0 instead.")
            ROADS[roadID].reverse_lanes()
            ROADS[roadID].shift_road() # no need for some roads
            ROADS[roadID].set_startintersection(edge.get("from"))
            ROADS[roadID].set_endintersection(edge.get("to"))
            INTERSECTIONS[edge.get("from")].insert_outroad(roadID)
            INTERSECTIONS[edge.get("to")].insert_inroad(roadID)


# ===== Parse junction part of SUMO roadnet (second step). =====
def parse_junction_2():
    global INTERSECTIONS, LANELINKS, LANE_NUMBER, INTERNAL_LANE_SHAPE
    for intersectionID in INTERSECTIONS:
        inRoads = INTERSECTIONS[intersectionID].get_inRoads()
        outRoads = INTERSECTIONS[intersectionID].get_outRoads()
        for inRoad in inRoads:
            for outRoad in outRoads:
                for connectionVia in LANELINKS:
                    if LANELINKS[connectionVia].get_startroad() == inRoad and LANELINKS[connectionVia].get_endroad() == outRoad:
                        roadLink = {}
                        if LANELINKS[connectionVia].get_direction() == 's':
                            roadLink["type"] = "go_straight"
                        elif LANELINKS[connectionVia].get_direction() == 'l':
                            roadLink["type"] = "turn_left"
                        elif LANELINKS[connectionVia].get_direction() == 'r':
                            roadLink["type"] = "turn_right"
                        else:
                            continue
                        roadLink["startRoad"] = LANELINKS[connectionVia].get_startroad()
                        roadLink["endRoad"] = LANELINKS[connectionVia].get_endroad()
                        roadLink["laneLinks"] = []
                        laneLink = {}
                        laneLink["startLaneIndex"] = LANE_NUMBER[roadLink["startRoad"]]-LANELINKS[connectionVia].get_startlaneindex()-1
                        laneLink["endLaneIndex"] = LANE_NUMBER[roadLink["endRoad"]]-LANELINKS[connectionVia].get_endlaneindex()-1
                        laneLinkShape = INTERNAL_LANE_SHAPE[connectionVia]
                        laneLink["points"] = parse_shape(laneLinkShape)
                        roadLink["laneLinks"].append(laneLink)
                        INTERSECTIONS[intersectionID].insert_roadlink(roadLink)
        INTERSECTIONS[intersectionID].set_default_traffic_light()


# ===== Initialize. =====
def init(selectedIntersectionDirectory, sumoRoadnetDirectory):
    load_selected_intersection(selectedIntersectionDirectory)
    load_sumo_roadnet(sumoRoadnetDirectory)


# ===== Parse SUMO roadnet. =====
def parse():
    record_lane_info()
    parse_connection()
    parse_junction_1()
    parse_edge()
    parse_junction_2()


# ===== Generate json file for City-Simulator roadnet. =====
def generate_roadnet(roadnetDirectory):
    global VIRTUAL, INTERSECTIONS, ROADS
    roadnet = {}
    roadnet["roads"] = []
    roadnet["intersections"] = []
    for roadID in ROADS:
        item = {}
        item["id"] = ROADS[roadID].id
        item["points"] = ROADS[roadID].points
        item["lanes"] = ROADS[roadID].lanes
        item["startIntersection"] = ROADS[roadID].startIntersection
        item["endIntersection"] = ROADS[roadID].endIntersection
        roadnet["roads"].append(item)
    for intersectionID in INTERSECTIONS:
        item = {}
        item["id"] = INTERSECTIONS[intersectionID].id
        item["point"] = INTERSECTIONS[intersectionID].point
        item["roads"] = []
        for road in INTERSECTIONS[intersectionID].get_inRoads():
            item["roads"].append(road)
        for road in INTERSECTIONS[intersectionID].get_outRoads():
            item["roads"].append(road)
        item["width"] = INTERSECTIONS[intersectionID].width
        item["roadLinks"] = INTERSECTIONS[intersectionID].roadLinks
        item["trafficLight"] = INTERSECTIONS[intersectionID].trafficLight
        if item["id"] in VIRTUAL:
            item["virtual"] = True
        else:
            item["virtual"] = False
        roadnet["intersections"].append(item)
    json.dump(roadnet, open(roadnetDirectory, 'w'), indent=2)


if __name__=="__main__":
    init("./selected-intersection", "./hz.net.xml")
    parse()
    generate_roadnet("./roadnet.json")
