# Short Description: This script should convert a video into individual frames

import ffmpeg
import cv2
from tqdm import tqdm
import os
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from ..apply_mask.apply_mask import apply_mask_single_img
from pathlib import Path


# Main loop
def main():
    args = setup_argparse()  # Setup arguments
    cameras = setup_cameras(args.data_path, args.debug_images)  # Setup list of camera objects

    # Go through each camera in list of cameras
    for camera in cameras:
        camera.videos_to_frames()  # Convert videos into written frames
    print('Completed!')


# Used for setting up argparse and returns the arguments, so they can be used for the rest of the code
def setup_argparse():
    '''
    Setting up argparse

    Returns:
        parse_args: Arguments from argparse
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-di', '--debug_images', help='Outputs deleted images to be used for debugging', type=bool, choices=[True], default=False)
    parser.add_argument('--data_path', help='Path to data', type=str, required=True)
    args = parser.parse_args()
    return args


# Sets up camera objects using the directory of cameras
def setup_cameras(data_path, debug_images, output_dir='./work_dir'):
    '''
    Inializes a list of camera objects

    Args:
        arguments (arg_parse): Arguments from argparse

    Returns:
        list: Camera objects
    '''
    cameras = []
    data_path = Path(data_path)
    camera_dirs = [str(name.name) for name in sorted(data_path.glob('*')) if data_path.is_dir()]
    for i in range(len(camera_dirs)):
        cameras.append(Camera(camera_name=camera_dirs[i], data_path=data_path, debug_images=debug_images, output_dir=output_dir))
    return cameras


# Video Class
class Video:
    '''Video class, contain information of each video'''
    def __init__(self, id, name, path, video_capture):
        self.id = id
        self.name = name
        self.path = path  # Absolute path to video file
        self.capturer = video_capture  # Video capturer to e.g. read frames from (cv2)
        self.num_frames = int(self.capturer.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video


# Camera Class
class Camera:
    '''Camera class, which contains information of the utilised cameras from the data directories'''
    def __init__(self, camera_name, data_path, debug_images, output_dir):
        self.name = camera_name
        self.input_dir = f'{data_path}/{camera_name}'  # Path to directory of input data
        self.output_dir = f'{output_dir}/{camera_name}/images/'  # Path to directory of where the new data (frames) should be written
        self.videos = self.setup_videos()  # Sets up a list of camera objects
        self.sar_width, self.sar_height = self.camera_sar_dimensions()  # SAR information, to have true info of the camera
        self.num_written_frames = 0  # A counter
        self.num_videos = len(self.videos)  # Number of videos in the camera

        self.debug_image = debug_images  # Arguments from argparse


    def videos_to_frames(self, mask_path, streamlit_progress=None):
        '''Extracts input videos into frames as output'''
        os.makedirs(self.output_dir)  # Creates new directory to store output

        # Creates directory if debug mode is activated
        if self.debug_image is True:
            os.mkdir(f'{self.output_dir}/debug/')

        # Timestamp
        time_stamp = 0

        # For each video in the list of video objects
        for video in self.videos:
            print(f'Camera: {self.name} | Video ({video.id+1}/{self.num_videos}): {video.name} | Image Size: [{self.sar_width}, {self.sar_height}, 3] | Output Directory: {self.output_dir}')

            # For each number of frame
            for i in tqdm(range(video.num_frames)):
                time_stamp += 1

                # Update streamlit progress
                if streamlit_progress:
                    streamlit_progress(i/video.num_frames)

                # Output directory
                temp_output_dir = self.output_dir  # Will change depending on a dark or gray frame is found AND if debug is activated

                # Loads frame
                retrieved, frame = video.capturer.read()  # Reads a frame
                if not retrieved:  # If no more frames being retrieved, then break loop
                    break

                # Checks if gray frames
                is_gray = self.check_grayscale_frames(frame)

                # Checks if dark frames
                is_dark = self.check_dark_frames(frame)

                # Prevents to write dark or gray frames
                if is_gray or is_dark:
                    if self.debug_image is True:
                        temp_output_dir = f'{temp_output_dir}/debug/'
                    else:
                        continue

                # Output frame
                video_name = re.split('[-.]', video.name)
                frame = cv2.resize(frame, (self.sar_width, self.sar_height), interpolation=cv2.INTER_AREA)

                # Apply Mask
                frame = apply_mask_single_img(frame, mask_path)

                # cv2.imwrite(f'{temp_output_dir}{self.count_frames()}_{video_name[1]}.jpg', frame)
                cv2.imwrite(f'{temp_output_dir}{time_stamp}_{video_name[1]}.jpg', frame)
                
            video.capturer.release()  # Releases utilised video

    def setup_videos(self):
        '''Creates a list of video objects for the camera, including video information

            Returns:
                list: Video objects
        '''
        video_paths = os.listdir(self.input_dir)  # Gets list of video paths in input directory
        video_paths = sorted(video_paths)
        videos = []

        # For each video path in list of video paths
        for idx, video_name in enumerate(video_paths):
            video_path = os.path.join(self.input_dir, video_name)
            video_capture = cv2.VideoCapture(video_path)  # cv2 Video capturer for e.g. reading frames
            video = Video(id=idx, name=video_name, path=video_path, video_capture=video_capture)  # Creates video
            videos.append(video)
        return videos

    def camera_sar_dimensions(self):
        '''Camera from the dataset contains errors in the meta-data, Sample Aspect Ratio (SAR) is then used to compute the true width and height of images.

            Returns:
                int: SAR image width
                int: Default image height
        '''
        # Get stream probe info of an input video
        probe_stream_info = ffmpeg.probe(self.videos[0].path)['streams'][0]

        # Gets the sample aspect ratio (SAR)
        sample_aspect_ratio = probe_stream_info['sample_aspect_ratio']
        sar_x, sar_y = sample_aspect_ratio.split(':')  # Splits the SAR ratio

        # Calculates the width of SAR
        sar_width = int(float(probe_stream_info['width']) * (float(sar_x)/float(sar_y)))
        image_height = int(probe_stream_info["height"])

        # Output SAR width and the image height
        # print(f'Image Size | Width: {sar_width}, Height: {image_height}')
        return sar_width, image_height

    def count_frames(self):
        ''' Increments number by one for each frame succesfully written. This is used for indexing image names.

            Returns:
                int: Number of frames

        '''
        self.num_written_frames += 1
        return self.num_written_frames

    def check_grayscale_frames(self, frame):
        ''' Checks if incomming frame is grayscaled instead of RGB

            Args:
                frame (cv2_image): A frame from a video

            Returns:
                bool: True if grayscaled, otherwise False
        '''
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converts RGB to HSV
        hist = cv2.calcHist([frame_gray], [1], None, [256], [0, 256])  # Using satuated values from HSV for histogram

        frequent_pixel_index = np.argmax(hist[1:])  # Returns the most frequent pixel, except for bin zero

        if frequent_pixel_index == 0:  # If the most frequent satuated value is index zero
            return True
        else:
            return False

    def check_dark_frames(self, frame):
        ''' Checks if incomming frame is dark due to night without any light

            Args:
                frame (cv2_image): A frame from a video

            Returns:
                bool: True if dark, otherwise False
        '''
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converts BGR to GRAY
        hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])  # Histogram

        total_pixels = np.sum(hist)
        dark_pixels = np.sum(hist[0:50])

        dark_ratio = dark_pixels / total_pixels

        if dark_ratio > 0.8:  # Dark ratio is above 80% of the pixels
            return True
        else:
            return False


if __name__ == "__main__":
    main()
