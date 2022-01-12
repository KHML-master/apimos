import argparse
import os
import json


def create_argparse():
    parser = argparse.ArgumentParser(description="Start inference")

    parser.add_argument(
        '--det',
        required=True,
        type=str,
        metavar='str',
        help='Path to the workspace directory of where the detector is located'
    )

    parser.add_argument(
        '--cls',
        required=True,
        type=str,
        metavar='str',
        help='Path to the workspace directory of where the classifier is located'
    )

    parser.add_argument(
        '--img',
        required=True,
        type=str,
        metavar='str',
        help='Path to image/images'
    )

    parser.add_argument(
        '--conf',
        required=False,
        type=float,
        metavar='float',
        default=0.5,
        help='Confidence for detection, default is 0.5'
    )

    return parser.parse_args()


def create_files(img_path, classifications):

    # Write File
    with open(img_path / 'classifications.json', 'w') as json_file:
        json.dump(classifications, json_file)


def get_files(path):
    try:
        return [path / f.name for f in sorted(path.glob('*'), key=os.path.getmtime)]
    except Exception:
        return [path]
