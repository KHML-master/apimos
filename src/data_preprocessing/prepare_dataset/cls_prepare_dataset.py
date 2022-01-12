# Save info such as position so we can go back and see how good it is to different positions
from tqdm import tqdm
import os
import json
import numpy as np
import PIL.Image
import re
# from utils.argparse import create_argparse
from . import utils


def get_anns_containing_pose(json_data):
    ann_list = []
    pose_categories = []
    for ann in json_data['annotations']:
        pose = ann['attributes']['pose']
        if pose == 'none':
            continue
        else:
            ann_list.append(ann)
            if not any(c['name'] == pose for c in pose_categories):
                category = {'id': len(pose_categories), 'name': pose}
                pose_categories.append(category)
    return ann_list, pose_categories


def construct_json(file_name, images, categories, output_dir):
    # Make dictionary
    json_data = {}
    json_data['info'] = {'dataset_name': 'classification dataset'}
    json_data['categories'] = categories
    json_data['images'] = [i for i in images]

    # Write File
    with open(f'{output_dir}/output/{file_name}.json', 'w') as json_file:
        json.dump(json_data, json_file)


def get_class_id(possible_categories, image_category):
    for pc in possible_categories:
        if image_category == pc.get('name'):
            return pc.get('id')


def get_roi_images(ann_list, pose_categories, json_images, data_dir, output_dir, save='False'):
    images = []
    for ann in tqdm(ann_list):
        image_id = ann['image_id']
        class_pose = ann['attributes']['pose']

        for img in json_images:
            if image_id == img['id']:
                _, dir_name, f_name, ext = re.split(r'/|\.', img['file_name'])

                file_name = f'{dir_name}_roi/i{f_name}_a{ann["id"]}.jpg'
                x, y, w, h = ann['bbox']

                if save:
                    # Open detection image
                    img_array = PIL.Image.open(f'{data_dir}/images/{dir_name}/{f_name}.{ext}')
                    img_array = np.array(img_array)

                    # Get new image based on ROI
                    img_array = img_array[int(y):int(y)+int(h), int(x):int(x)+int(w)]

                    # Make dir if not exist
                    if os.path.isdir(f'{output_dir}/output/cls_images/{dir_name}_roi') is False:
                        os.mkdir(f'{output_dir}/output/cls_images/{dir_name}_roi')

                    # Save Image
                    img_array = PIL.Image.fromarray(img_array, 'RGB')
                    img_array.save(f'{output_dir}/output/cls_images/{file_name}')

                class_id = get_class_id(pose_categories, class_pose)
                image = {'file_name': file_name, 'category_id': class_id, 'roi_x': x, 'roi_y': y, 'roi_width': w, 'roi_height': h, 'original_width': img['width'], 'original_height': img['height'], 'area': ann['area']}
                images.append(image)

            else:
                continue
    return images


def get_classes(directory, data_dir):
    # Read JSON
    with open(f'{data_dir}/annotations/{directory["name"]}.json', 'r') as f:
        json_data = json.load(f)

    # Get Annotations that do not contain a pose label
    _, pose_categories = get_anns_containing_pose(json_data)

    return pose_categories


def setup_output_directory(output_dir, save):
    if os.path.isdir(f'{output_dir}/output/') is False:
        os.mkdir(f'{output_dir}/output/')

    if save:
        if os.path.isdir(f'{output_dir}/output/cls_images/') is False:
            os.mkdir(f'{output_dir}/output/cls_images/')


if __name__ == '__main__':
    argparse = utils.cls_create_argparse()

    setup_output_directory(argparse.output, argparse.save)

    with open(argparse.config, 'r') as f:
        config = json.load(f)

    # Get possible classes from first directory (Can be an issue, but works for now)
    pose_classes = get_classes(config['directories'][0], argparse.data)

    train_data = []
    val_data = []
    test_data = []

    for directory in config['directories']:
        print(directory['name'])

        # Read JSON
        with open(f'{argparse.data}/annotations/{directory["name"]}.json', 'r') as f:
            json_data = json.load(f)

        # Get Annotations that do not contain a pose label
        ann_list, _ = get_anns_containing_pose(json_data)

        # Images
        json_images = json_data['images']

        # Get RoI images
        images = get_roi_images(ann_list, pose_classes, json_images, argparse.data, argparse.output, argparse.save)

        # Shuffle
        np.random.shuffle(images)

        # Select and split
        train_split = directory['split_values']['train']
        val_split = directory['split_values']['validation']
        select = np.split(images, [int(len(images) * train_split), int(len(images) * (train_split+val_split))])

        train_data.extend(select[0])
        val_data.extend(select[1])
        test_data.extend(select[2])

    # Save files
    construct_json('train', train_data, pose_classes, argparse.output)
    construct_json('valid', val_data, pose_classes, argparse.output)
    construct_json('test', test_data, pose_classes, argparse.output)
