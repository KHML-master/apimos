from collections import namedtuple
import argparse
import pickle
import json


def read_file(file_path):
    try:
        with open(f'{file_path}', 'rb') as f:
            extension = file_path.split('.')[-1]
            if extension == 'pkl':
                return pickle.load(f)
            elif extension == 'json':
                return json.load(f)
    except Exception:
        return None


def create_argparse():
    parser = argparse.ArgumentParser(description="Settings for visualize results using centerpoints")

    parser.add_argument(
        '-ip', '--image_path',
        type=str,
        metavar='str',
        required=True,
        help='Path to where the images are located'
    )

    parser.add_argument(
        '-jp', '--json_path',
        type=str,
        metavar='str',
        required=True,
        help=''
    )

    parser.add_argument(
        '-pp', '--pickle_path',
        type=str,
        metavar='str',
        required=False,
        help='Path to where the file containing bboxes (either json/pickle)'
    )

    parser.add_argument(
        '-op', '--output_path',
        type=str,
        metavar='str',
        required=True
    )

    parser.add_argument(
        '-es', '--ellipse_scale',
        type=float,
        metavar='float',
        required=False,
        default=0.9
    )
    parser.add_argument(
        '-th', '--threshold',
        type=float,
        metavar='float',
        required=False,
        default=0)

    return parser.parse_args()


def save_images(images, image_paths, output_path):
    # Calculate center point for each bounding box
    # Draw center point with color = accuracy
    for img, img_path in zip(images, image_paths):
        img.save(output_path + '/' + img_path.split('/')[-1])


# Def: accuracy to color
def acc_to_color(acc):
    if acc >= 0.5:
        R = 255 * ((1-acc)*2)
        G = 255
    elif acc < 0.5:
        R = 255
        G = 255*(acc*2)
    return (int(R), int(G), 0)


def acc_to_color_interp(acc):
    acc = acc * 255

    R = 255 - acc
    G = 0 + acc

    return (int(R), int(G), 0)


def collect_bboxes(det_image):
    bboxes = []
    BoundingBox = namedtuple('BoundingBox', ['pred_class', 'pred_score', 'x', 'y', 'w', 'h', 'det_pred'])
    for cls_roi in det_image:
        bbox = BoundingBox(cls_roi['pred_class'], float(cls_roi['pred_score']), float(cls_roi['x']), float(cls_roi['y']), float(cls_roi['w']), float(cls_roi['h']), float(cls_roi['detection_confidence']))
        bboxes.append(bbox)
    return bboxes


def get_coordinates(bbox, scale):
    Coordinates = namedtuple('Coordinates', ['x0', 'y0', 'x1', 'y1', 'xm', 'ym'])

    try:
        x0 = bbox.x/2 + (bbox.w * scale / 2)
        y0 = bbox.y/2 + (bbox.h * scale / 2)
        x1 = (bbox.x/2 + bbox.w) - (bbox.w * scale / 2)
        y1 = (bbox.y/2 + bbox.h) - (bbox.h * scale / 2)
    except Exception:
        x0 = bbox[0]/2 + (bbox[2] * scale / 2)
        y0 = bbox[1]/2 + (bbox[3] * scale / 2)
        x1 = (bbox[0]/2 + bbox[2]) - (bbox[2] * scale / 2)
        y1 = (bbox[1]/2 + bbox[3]) - (bbox[3] * scale / 2)

    # Classification Text
    xm = x0 + (abs(x0 - x1))
    ym = y0 + (abs(y0 - y1))

    return Coordinates(x0, y0, x1, y1, xm, ym)
