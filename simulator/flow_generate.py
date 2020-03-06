import random
import re
import test.set_sumo_home as st

st.set_sumo_home()
import xml.etree.ElementTree as ET


def generate(route_path):
    tree = ET.parse('./roadnet/{}.net.xml'.format(route_path))
    root = tree.getroot()
    edgeid = []
    for edge in root.findall("edge"):
        if "function" not in edge.attrib.keys():
            if edge.get("type") == "highway.primary":
                edgeid.append(edge.get('id'))

    with open('./flow/{}.rou.xml'.format(route_path), 'w') as routes:
        count = 1
        for i in range(0, 3600):
            edgeid_comb = random.sample(edgeid, 2)
            edge_start, edge_end = re.sub(":", "", edgeid_comb[0]), re.sub(":", "", edgeid_comb[1])
            if count % 2 == 1:
                print(
                    "<flow id=\"%d\" type=\"SUMO_DEFAULT_TYPE\" begin=\"0\" end=\"3600\" number=\"1200\" from=\"%s\" to=\"%s\" departPos=\"0\" departLane=\"best\" departSpeed=\"5.0\"/>" % (
                        i, edge_start, edge_end), file=routes)
                count += 1
            else:
                print(
                    "<flow id=\"%d\" type=\"SUMO_DEFAULT_TYPE\" begin=\"0\" end=\"3600\" number=\"800\" from=\"%s\" to=\"%s\" departPos=\"0\" departLane=\"best\" departSpeed=\"5.0\"/>" % (
                        i, edge_start, edge_end), file=routes)
                count += 1


if __name__ == '__main__':
    generate('binjiang0')
