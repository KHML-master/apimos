import xml.etree.ElementTree as ET
from collections import namedtuple

def invert_axis(points, max_value):
    inverted_points = []
    for point in points:
        inverted_points.append(max_value-point)
    return inverted_points

def movePointToPen(points, width, length):
    new_points = []
    for point in points:
        if point[0] < 0:
            point[0] = 0
        elif point[0] > width:
            point[0] = width
        if point[1] < 0:
            point[1] = 0
        elif point[1] > length:
            point[1] = length
        new_points.append(point)
    return new_points

def laying_specs(image):
    pigs_df = image.copy()
    pigs_df = pigs_df[(pigs_df['pose'] != 'standing') & (pigs_df['pose'] != 'sitting')]
    position_idx = pigs_df['position_idx'].mean()
    pigs_df = pigs_df[pigs_df['position_idx'] > 0.5]
    return position_idx, len(pigs_df)

def get_points_xml(xml_path):
    xml_file = ET.parse(xml_path)
    root = xml_file.getroot()
    pts_src = []
    pts_dst = []

    for point in root[2]:
        pts_src.append([float(item) for item in point.attrib['points'].split(',')])
        pts_dst.append([float((point[1].text)), float((point[0].text))])

    # Line for visualisation
    lines = []
    for p in pts_dst:
        lines.append(593-p[1])
    p_x = []
    p_y = []
    for p in pts_src:
        p_x.append(p[0])
        p_y.append(p[1])
    
    points = namedtuple('Point', 'lines xs ys')
    return points(lines, p_x, p_y)

def get_points_json(xml_path):
    xml_file = ET.parse(xml_path)
    root = xml_file.getroot()
    pts_src = []
    pts_dst = []

    for point in root[2]:
        pts_src.append([float(item) for item in point.attrib['points'].split(',')])
        pts_dst.append([float((point[1].text)), float((point[0].text))])

    # Line for visualisation
    lines = []
    for p in pts_dst:
        lines.append(593-p[1])
    p_x = []
    p_y = []
    for p in pts_src:
        p_x.append(p[0])
        p_y.append(p[1])
    
    points = namedtuple('Point', 'lines xs ys')
    return points(lines, p_x, p_y)