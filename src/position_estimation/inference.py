import cv2 as cv
import json
import numpy as np
from tqdm import tqdm


def load_file(file_path):
    with open(f'{file_path}', 'rb') as f:
        return json.load(f)


def extract_bboxes(image):
    bboxes = []
    for cls_roi in image:
        x0 = cls_roi['x']
        y0 = cls_roi['y']
        # x1 = cls_roi['x'] + cls_roi['w']
        # y1 = cls_roi['y'] + cls_roi['h']

        c = float(y0) + float(cls_roi['h']) / 2

        bbox = [float(x0), c]
        bboxes.append(bbox)
    return bboxes


def transform_bboxes(original_bboxes, cam):
    try:
        tbb = cv.perspectiveTransform(np.array(original_bboxes).reshape(len(original_bboxes), 1, -1), np.load(f'./position_estimation/data_calibration/cal_matrix/{cam}.npy'))
    except Exception:
        print('No bounding boxes in image.')
        return []
    return tbb


def transform_bbox(bbox, cam):
    x0, y0, x1, y1, _ = bbox

    cx = float(x0) + (float((x1-x0)) / 2)
    cy = float(y1)

    tbb = cv.perspectiveTransform(np.array([[cx, cy]]).reshape(1, 1, -1), np.load(f'./src/position_estimation/data_calibration/cal_matrix/{cam}.npy'))
    
    t = np.load(f'./src/position_estimation/data_calibration/cal_matrix/{cam}.npy')

    return tbb

def transform_bbox4points(bbox, cam):
    x0, y0, x1, y1 = bbox

    cx = float(x0) + (float((x1)) / 2)
    cy = float(y0+y1)

    tbb = cv.perspectiveTransform(np.array([[cx, cy]]).reshape(1, 1, -1), np.load(f'./pig_vision/position_estimation/data_calibration/cal_matrix/{cam}.npy'))
    return tbb


def append_world_points_to_json(json_image, t_bboxes):
    for json_cls, tb in zip(json_image, t_bboxes):
        json_cls['world_x'] = tb[0][0]
        json_cls['world_y'] = tb[0][1]


if __name__ == '__main__':
    json_file = load_file('position_estimation/classifications.json')
    for image in tqdm(json_file):
        try:
            image_name = image[0]['detection_image'][1]

            original_bboxes = extract_bboxes(image)

            t_bboxes = transform_bboxes(original_bboxes)

            append_world_points_to_json(image, t_bboxes)

        except Exception:
            print('Empty Image')

    with open('world_points.json', 'w') as f:
        json.dump(json_file, f)
