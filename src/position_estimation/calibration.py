import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import os

from pathlib import Path
import json


class CalibrationXML:
    def __init__(self, save_path, xml_path, cam_name):
        self.save_path = save_path
        self.cam_name = cam_name
        self.pts_src, self.pts_dst = self.get_points_xml(xml_path)

    def calibrate_camera(self):

        h, status = cv.findHomography(self.pts_src, self.pts_dst)
        print('status: ', status)
        self.save_matrix(h)

    def save_matrix(self, h):
        print('Saving matrix')
        np.save(self.save_path+'/'+self.cam_name, h)

    def get_points_xml(self, xml_path):
        xml_file = ET.parse(xml_path)
        root = xml_file.getroot()
        pts_src = []
        pts_dst = []
        for point in root[2]:
            pts_src.append(np.float32([float(item) for item in point.attrib['points'].split(',')]))
            pts_dst.append(np.float32([float((point[1].text)), float((point[0].text))]))

        return np.array(pts_src).reshape(len(pts_src), 1, -1), np.array(pts_dst).reshape(len(pts_dst), 1, -1)

    
class CalibrationJSON:
    def __init__(self, config):
        self.config = config
        self.camera_name = config['camera']['name']

        world_pts = config['world points']
        self.pts_dst = {'pt1': json.loads(world_pts['pt1']),
                        'pt2': json.loads(world_pts['pt2']),
                        'pt3': json.loads(world_pts['pt3']),
                        'pt4': json.loads(world_pts['pt4']),
                        'pt5': json.loads(world_pts['pt5']),
                        'pt6': json.loads(world_pts['pt6']),
                        'pt7': json.loads(world_pts['pt7']),
                        'pt8': json.loads(world_pts['pt8'])}
    
    def calibrate_camera(self, pts_src):
        pts_src = list(pts_src.values())
        pts_dst = list(self.pts_dst.values())
        
        pts_src_arr = np.array(pts_src).reshape(len(pts_src), 1, -1)
        pts_dst_arr = np.array(pts_dst).reshape(len(pts_dst), 1, -1)

        homography_mat, status = cv.findHomography(pts_src_arr, pts_dst_arr)

        save_path = f'../../src/position_estimation/data_calibration/cal_matrix/{self.camera_name}.npy'

        print(f'[INFO] Saved Homography Matrix to: {save_path}')
        np.save(save_path, homography_mat)
    
if __name__ == '__main__':
    path = 'data_calibration/xml_files'
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            c = CalibrationXML('data_calibration/cal_matrix', path+'/'+filename, filename.split('.')[0])
            c.calibrate_camera()
