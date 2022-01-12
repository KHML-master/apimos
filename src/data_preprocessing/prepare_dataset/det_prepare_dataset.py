# Input: JSON files: one for each camera
# dataset_seges_small/
#   cam001/
#       original/
#       mask/
#   cam002/
#       original/
#       mask/


# Referencers (Pushed to Github)
#   test.json
#   train.json
#   validation.json

# Class: Create an object for each camera containing:
#   JSON data
#   annotated_images
#   load configs for camera

# For each camera:
#   For each Images:
#       Check if camera has annotations
#       If not, delete from JSON data
#       else annotated_image += 1


# Class: Create dataset object:
#   Load global_config values
#   data_mode = [train, val, test]

# def: build dataset:

# For each data_mode:
#   For each camera:
#       select data based on config
#    Output: JSON File for that datamode

# Overwrite with new_jsons

# Ouput: 3 JSON files: Train, Val, Test
# Ouput: Datafolder
import numpy as np
# from PIL import Image, ImageDraw
# import os
from . import utils


class Dataset():
    def __init__(self, name, output_path):
        self.json_dict = {}
        self.name = name
        self.output_path = output_path

    def build(self, cameras, seed):
        np.random.seed(seed)

        temp_train_img = []
        temp_validation_img = []
        temp_test_img = []

        temp_train_ann = []
        temp_validation_ann = []
        temp_test_ann = []

        # Split images and annotations into three groups of train, validation and test
        for cam in cameras:
            # Random pick percent of images from camera
            train_split = cam.split_values['train']
            val_split = cam.split_values['validation']

            # Get JSONs
            json_data_img = cam.json_data['images']
            json_data_ann = cam.json_data['annotations']

            # Shuffle Images
            np.random.shuffle(json_data_img)

            # Selecting Images
            select = np.split(json_data_img, [int(len(json_data_img)*train_split), int(len(json_data_img)*(train_split+val_split))])
            temp_train_img.extend(select[0])
            temp_validation_img.extend(select[1])
            temp_test_img.extend(select[2])

            # Select Annotations TODO: Maybe unhardcode (Make it to a function)
            train_img_id, validation_img_id, test_img_id = self.get_ids(select)

            # Extends lists
            temp_train_ann.extend([ann for ann in json_data_ann if ann['image_id'] in train_img_id])
            temp_validation_ann.extend([ann for ann in json_data_ann if ann['image_id'] in validation_img_id])
            temp_test_ann.extend([ann for ann in json_data_ann if ann['image_id'] in test_img_id])

        # Data Modes
        train_mode = {'mode_type': 'train', 'images': temp_train_img, 'anns': temp_train_ann}
        validation_mode = {'mode_type': 'validation', 'images': temp_validation_img, 'anns': temp_validation_ann}
        test_mode = {'mode_type': 'test', 'images': temp_test_img, 'anns': temp_test_ann}

        # Save each JSON mode (train, validation, test)
        info_dict = {'dataset_name': self.name}
        for mode in [train_mode, validation_mode, test_mode]:
            mode_type, imgs, anns = mode.values()

            # Combine string (fixing error of string being splitted. There might be a better solution)
            mode_type = ''.join(mode_type)  # Test if it works commented out

            # Re-name file_name for each img and ann (file path)
            # img_name = f'{cam_name}_{image_name}'

            # Construct JSON mode file
            json_construction = utils.construct_json(info_dict, imgs, anns, cameras[0].json_data['categories'])

            # Save JSON
            utils.save_json(f'{self.output_path}/{mode_type}', json_construction)

    def get_ids(self, selected):
        img_id_list = []
        for select in selected:
            temp_img_ids = []
            for img in select:
                temp_img_ids.append(img['id'])
            img_id_list.append(temp_img_ids)
        return img_id_list


class Camera:
    def __init__(self, config, data_input_path, output_path):
        self.name = config['name']
        self.mode = config['mode']
        self.split_values = config['split_values']
        self.ann_images = 0
        self.directory_path = f'{data_input_path}/{self.name}'
        # self.json_data = json_utils.load_json(path=f'{self.directory_path}/annotations/instances_default.json')
        self.json_data = utils.load_json(path=f'{data_input_path}/annotations/{self.name}.json')
        self.example_image_path = self.json_data['images'][0]["file_name"]
        self.output_path = output_path
        self.ann_ids = []


def sort_change_json(cameras, mask):
    for cam in cameras:

        # Annotations
        ann_ids = []
        for ann in cam.json_data['annotations']:
            ann_ids.append(ann['image_id'])

        unique, counts = np.unique(ann_ids, return_counts=True)
        ann_ids_dict = dict(zip(unique, counts))

        # Images
        temp_images = []
        for img in cam.json_data['images']:
            if img['id'] in unique:
                del img['license'], img['flickr_url'], img['coco_url'], img['date_captured']
                img['bbox_numbers'] = str(ann_ids_dict[img['id']])
                img['cam'] = cam.name
                t_path = img['file_name'].split('/')
                if mask:
                    img['file_name'] = f'{t_path[-2]}_masked/{t_path[-1]}'
                else:
                    img['file_name'] = f'{t_path[-2]}/{t_path[-1]}'
                temp_images.append(img)

        cam.json_data['images'] = temp_images

        cam.ann_images = len(cam.json_data['images'])
        cam.ann_ids = ann_ids


def change_id(cameras):
    new_img_id = 0
    new_ann_id = 0

    for cam in cameras:
        map_dict = {}

        # Images
        for img in cam.json_data['images']:
            img_id = img['id']
            map_dict[img_id] = new_img_id
            img['id'] = new_img_id
            new_img_id += 1

        # Annotations
        for ann in cam.json_data['annotations']:
            ann['id'] = new_ann_id
            ann['image_id'] = map_dict[ann['image_id']]
            new_ann_id += 1


if __name__ == '__main__':
    # Argparse
    argparser = utils.create_argparse()

    # Load Config
    config_data = utils.load_json(path=argparser.config)

    # Create Camera for each directory in config file
    cameras = []
    for directory in config_data['directories']:
        cameras.append(Camera(directory, argparser.data, argparser.output))

    # Check which images are labeled to prevent using images that are not labeled
    sort_change_json(cameras, argparser.mask)

    # Change id to make each image and ann to make them unique
    change_id(cameras)

    dataset = Dataset(argparser.name, argparser.output)
    dataset.build(cameras, argparser.seed)

    print('[INFO] Dataset is now prepared!')
