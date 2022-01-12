from PIL import Image
from . import masking
#import masking
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse


def create_argparse():
    parser = argparse.ArgumentParser(description="Settings for Masking")

    parser.add_argument('--input',
                        required=True,
                        type=str,
                        metavar='str',
                        help='Path to directory containing cameras of images'
                        )

    return parser.parse_args()


def apply_mask(images, camera_dir, mask_path, mask_dir):
    for img in tqdm(images):
        image_path = f'{camera_dir}/{img}'

        # Original Image
        original_img = Image.open(image_path).convert('RGB')
        width, height = original_img.size

        # Background Image
        background = Image.new('RGB', (width, height), (0, 0, 0))

        # Mask
        mask = Image.open(mask_path).convert('L')

        # Applying mask to original image
        output = Image.composite(original_img, background, mask)

        # Saving the output
        output.save(f'{mask_dir}/{img}')


def apply_mask_single_img(image, mask_path):
    # image_path = f'{camera_dir}/{img}'

    # Original Image
    original_img = Image.fromarray(image).convert('RGB')
    width, height = original_img.size

    # Background Image
    background = Image.new('RGB', (width, height), (0, 0, 0))

    # Mask
    mask = Image.open(mask_path).convert('L')

    # Applying mask to original image
    output = Image.composite(original_img, background, mask)

    return np.array(output)


def each_camera2(dataset_dir):

    # Each Camera
    cameras = glob(f'{dataset_dir}/images/*[!masked]')
    for cam in cameras:  # Change value

        # Get all images in camera
        images = os.listdir(cam)

        # Mask Directory of Images
        mask_image_dir = f'{cam}_masked'
        if os.path.isdir(mask_image_dir) is False:
            os.mkdir(mask_image_dir)

        # Making Mask
        example_image_name = images[0]
        image_absolute_path = f'{cam}/{example_image_name}'
        mask_path = masking.create_mask(dataset_dir, image_absolute_path, cam)

        # Applying Mask to images
        apply_mask(images, cam, mask_path, mask_image_dir)


def mask_single_directory(camera_dir):
    # Get all images in camera
    images = os.listdir(camera_dir)

    # Mask Directory of Images
    mask_image_dir = f'{camera_dir}_masked'
    if os.path.isdir(mask_image_dir) is False:
        os.mkdir(mask_image_dir)

    # Making Mask
    example_image_name = images[0]
    image_absolute_path = f'{camera_dir}/{example_image_name}'
    dataset_dir = f'{camera_dir}/../..'  # Another approach should be preffered
    mask_path = masking.create_mask(dataset_dir, image_absolute_path, camera_dir)

    # Applying Mask to images
    apply_mask(images, camera_dir, mask_path, mask_image_dir)


if __name__ == '__main__':
    argparser = create_argparse()
    mask_single_directory(argparser.input)
    # each_camera(argparser.input)
