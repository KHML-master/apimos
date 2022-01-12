import argparse
import json


def cls_create_argparse():
    parser = argparse.ArgumentParser(description="Settings for preparing dataset")
    parser.add_argument(
        '--data',
        required=True,
        type=str,
        metavar='str',
        help='Path to the directory containing data'
    )

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        metavar='str',
        help='Path to the config that contains information of settings for each camera'
    )

    parser.add_argument(
        '--output',
        required=True,
        type=str,
        metavar='str',
        help='Path to where output should be located'
    )

    parser.add_argument(
        '--save',
        # required=False,
        # type=bool,
        # metavar='bool',
        default=False,
        action='store_true',
        help='Save region of interest images ("usage: --save" not "--save BOOL")'
    )

    return parser.parse_args()


def det_create_argparse():
    parser = argparse.ArgumentParser(description="Settings for preparing dataset")

    parser.add_argument(
        '--name',
        required=True,
        type=str,
        metavar='str',
        help='Name of the dataset'
    )

    parser.add_argument(
        '--data',
        required=True,
        type=str,
        metavar='str',
        help='Path to the directory containing data'
    )

    parser.add_argument(
        '--output',
        required=True,
        type=str,
        metavar='str',
        help='Path to the directory where the output should be placed'
    )

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        metavar='str',
        help='Path to the config that contains information of settings for each camera'
    )

    parser.add_argument(
        '--seed',
        required=False,
        type=int,
        metavar='int',
        help='The seed that should be used for shuffling the dataset, default is 123',
        default=123
    )

    parser.add_argument(
        '--mask',
        required=False,
        type=bool,
        metavar='int',
        help='If to apply mask to images',
        default=False
    )

    return parser.parse_args()


def save_json(name, data):
    with open(f'{name}.json', 'w') as json_file:
        json.dump(data, json_file)


def load_json(path):
    # with open(path, 'r') as f:
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


def construct_json(info, img, ann, cat):
    json_data = {}
    json_data['info'] = info
    json_data['categories'] = cat
    json_data['images'] = img
    json_data['annotations'] = ann

    return json_data
