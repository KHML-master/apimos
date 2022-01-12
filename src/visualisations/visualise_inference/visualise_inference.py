import json
import pandas as pd
from tqdm import tqdm
import os
from . import utils
from . import tools
from pathlib import Path

class PenVideo():
    def __init__(self, JSON_path, input_dir, config):
        self.pen_length = float(config['pen_info']['length'])
        self.pen_width = float(config['pen_info']['width'])

        self.output_dir = input_dir / 'analysis'
        # Video name
        self.video_name = -1
        # Camera
        self.camera = -1
        # dict of video_frames
        self.video_frames, self.meta_data = self.loadData(JSON_path)

    def loadData(self, JSON_path):
        # For each images create a video_frame()
        temp_video_images = pd.DataFrame()
        temp_meta_data = pd.DataFrame()

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

            with open(f'{JSON_path}', 'rb') as f:
                file = json.load(f)

            for image in tqdm(file):
                if len(image) > 0:
                    image_index = int(Path(image[0]['detection_image']).stem.split('_')[0])
                    image_name = image[0]['detection_image']

                    temp_df = self.createPigDataframe(image, image_index)
                    temp_video_images = temp_video_images.append(temp_df)

                    num_sternal = len(temp_df[temp_df['pose'] == 'sternal'])
                    num_lateral = len(temp_df[temp_df['pose'] == 'lateral'])
                    num_standing = len(temp_df[temp_df['pose'] == 'standing'])
                    num_sitting = len(temp_df[temp_df['pose'] == 'sitting'])

                    laying_idx, laying_in_spalte = utils.laying_specs(temp_df)

                    temp_meta_data = temp_meta_data.append(
                        {   
                            'image_name': image_name,
                            'image_index': image_index,
                            'num_lateral': num_lateral,
                            'num_sternal': num_sternal,
                            'num_standing': num_standing,
                            'num_sitting': num_sitting,
                            'detected_pigs': len(image),
                            'mean_position_idx': laying_idx,
                            'laying_in_spalte': laying_in_spalte
                        },
                        ignore_index=True
                    )

            # Save Dataframes
            self.save_dataframes(temp_meta_data, temp_video_images)
        else:
            print('Analysis directory already exists for this project \n Loading files')
            temp_video_images = pd.read_csv(self.output_dir/'video_frames.csv')
            temp_meta_data = pd.read_csv(self.output_dir/'overview.csv')

        return temp_video_images, temp_meta_data

    def save_dataframes(self, meta_data, video_frames):
        meta_data.to_csv(self.output_dir / 'overview.csv')
        video_frames.to_csv(self.output_dir / 'video_frames.csv')

    def createPigDataframe(self, image, image_index):
        temp_df = pd.DataFrame()
        for bbox in image:

            temp_df = temp_df.append({
                'image_index': image_index,
                'position_idx': utils.movePointToPen([[
                    float(bbox['world_x']),
                    float(bbox['world_y'])
                    ]],
                    width=self.pen_width,
                    length=self.pen_length,
                    )[0][1]/self.pen_length,
                'world_x': float(bbox['world_x']),
                'world_y': float(bbox['world_y']),
                'bbox_x0': float(bbox['x0']),
                'bbox_y0': float(bbox['y0']),
                'bbox_x1': float(bbox['x1']),
                'bbox_y1': float(bbox['y1']),
                'pose': bbox['pred_class'],
                'det_conf': float(bbox['detection_confidence']),
                'pose_conf': bbox['pred_score'],
            },
                ignore_index=True)

        return temp_df


def inference(config, cam, input_dir):
    if not config.getboolean('visualise', 'enable'):
        return

    # Create analysis dir
    analysis_dir = input_dir / cam / 'analysis'
    if not analysis_dir.is_dir():
        os.makedirs(analysis_dir)

    video = PenVideo(input_dir / cam / 'classifications.json', input_dir / cam / 'analysis', config)

    # Use bounding boxes
    if config.getboolean('visualise', 'bounding_boxes'):
        tools.BoundingBoxes(video, input_dir, analysis_dir)
    
    # Use positions in pen
    if config.getboolean('visualise', 'position_in_pen'):
        tools.PositionInPen(video, input_dir, cam, analysis_dir)

    # Use poses per day
    if config.getboolean('visualise', 'poses_per_day'):
        tools.PosesPerDay(video, input_dir, analysis_dir)

def main(input_dir, cam, config):
    video = PenVideo(input_dir / cam / 'classifications.json', input_dir, config)
    return video
