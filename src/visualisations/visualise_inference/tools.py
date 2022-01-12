import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from . import utils
import random
from pathlib import Path
import os
from tqdm import tqdm
import pickle

class Tool:
    def __init__(self, analysis_dir):
        self.analysis_dir = analysis_dir

    def create_dir(self, tool_name):
        # Create dir
        output_dir = self.analysis_dir/tool_name
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        else:
            print('[WARNING] Be aware that an analysis directory already exists for this project \n You might overwrite some files')
        return output_dir

class PositionInPen(Tool):
    def __init__(self, penVideo, input_dir, cam, analysis_dir):
        super().__init__(analysis_dir)
        print('[INFO] Using Visualising tool: Position In Pen')
        self.output_dir = self.create_dir('position_in_pen')
        self.process(penVideo, input_dir)

    def process(self, penVideo, input_dir):
        # Go through each frame
        for meta_data in tqdm(penVideo.meta_data.iterrows(), total=penVideo.meta_data.shape[0]):
            meta_data = meta_data[1]

            image_idx = int(meta_data['image_index'])

            image_df = penVideo.video_frames.loc[penVideo.video_frames['image_index'] == image_idx].copy()
            
            # Move points to within the pen
            image_df.loc[image_df.world_x < 0, 'world_x'] = 0
            image_df.loc[image_df.world_x > 200, 'world_x'] = 200
            image_df.loc[image_df.world_y < 0, 'world_y'] = 0
            image_df.loc[image_df.world_y > 593, 'world_y'] = 593

            # Invert axis
            image_df['world_y'] = 593-image_df['world_y']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
            sns.scatterplot(ax=ax1, x='world_x', y='world_y', hue='pose', s=100, data=image_df)
            ax1.legend(bbox_to_anchor=(-0.5, 1), loc=2, borderaxespad=0.)
            ax1.set(ylim=(-20, 620), xlim=(-20, 220))
            ax1.set_aspect(1)

            img_name = penVideo.meta_data.loc[penVideo.meta_data['image_index'] == image_idx]['image_name'].values[0]

            img = Image.open(input_dir/img_name)
            ax2.imshow(img, interpolation='nearest')
            #ax2.plot((self.points.xs[0], self.points.xs[1]), (self.points.ys[0], self.points.ys[1]), color='red')
            #ax2.plot((self.points.xs[2], self.points.xs[3]), (self.points.ys[2], self.points.ys[3]), color='blue')
            #ax2.plot((self.points.xs[4], self.points.xs[5]), (self.points.ys[4], self.points.ys[5]), color='green')
            #ax2.plot((self.points.xs[6], self.points.xs[7]), (self.points.ys[6], self.points.ys[7]), color='yellow')

            #ax1.hlines(self.points.lines[0], 0, 200, color=['red'])
            #ax1.hlines(self.points.lines[2], 0, 200, color=['blue'])
            #ax1.hlines(self.points.lines[4], 0, 200, color=['green'])
            #ax1.hlines(self.points.lines[6], 0, 200, color=['yellow'])

            # Save img
            image_name = Path(meta_data.image_name).name
            plt.savefig(self.output_dir/image_name)
            plt.cla()
            plt.clf()
            plt.close()


class BoundingBoxes(Tool):
    def __init__(self, penVideo, input_dir, analysis_dir):
        super().__init__(analysis_dir)
        print('[INFO] Using Visualising tool: Bounding Boxes')
        self.output_dir = self.create_dir('bounding_boxes')
        self.process(penVideo, input_dir)

    def process(self, penVideo, input_dir):
        # Go through each frame
        for meta_data in tqdm(penVideo.meta_data.iterrows(), total=penVideo.meta_data.shape[0]):
            meta_data = meta_data[1]
            image_name = Path(meta_data.image_name).name

            img = Image.open(input_dir / meta_data.image_name)
            img = img.convert('HSV')
            
            draw = ImageDraw.Draw(img, mode='HSV')

            image = penVideo.video_frames.loc[penVideo.video_frames['image_index'] == int(meta_data.image_index)].copy()

            desired = image.loc[image['det_conf'] > 0.5]

            bbox_array = np.array((
                desired['bbox_x0'].values,
                desired['bbox_y0'].values,
                desired['bbox_x1'].values,
                desired['bbox_y1'].values
                ))

            bbox_array = np.column_stack(bbox_array)

            for bbox in bbox_array:
                x0 = bbox[0]
                y0 = bbox[1]
                x1 = bbox[2]
                y1 = bbox[3]

                draw.rectangle(
                    [(x0, y0), (x1, y1)],
                    outline=(random.randint(0, 360), random.randint(190, 230), 255),
                    width=3
                )

            img = img.convert('RGB')
            img.save(self.output_dir/image_name)


class PosesPerDay(Tool):
    def __init__(self, penVideo, input_dir, analysis_dir):
        super().__init__(analysis_dir)
        print('[INFO] Using Visualising tool: Poses Per Day')
        self.output_dir = self.create_dir('day_visualise')
        self.process(penVideo)

    def process(self, penVideo):
        pose_dict = {'num_lateral': [], 
                     'num_sternal': [],
                     'num_standing': [],
                     'num_sitting': []}

        for data in tqdm(penVideo.meta_data.iterrows(), total=penVideo.meta_data.shape[0]):
            pose_dict['num_lateral'].append(int(data[1].num_lateral))
            pose_dict['num_sternal'].append(int(data[1].num_sternal))
            pose_dict['num_standing'].append(int(data[1].num_standing))
            pose_dict['num_sitting'].append(int(data[1].num_sitting))

        pose_dict['num_lateral'] = [sum(pose_dict['num_lateral'][i:i+10])/10 for i in range(0, len(pose_dict['num_lateral']), 10)]
        pose_dict['num_sternal'] = [sum(pose_dict['num_sternal'][i:i+10])/10 for i in range(0, len(pose_dict['num_sternal']), 10)]
        pose_dict['num_standing'] = [sum(pose_dict['num_standing'][i:i+10])/10 for i in range(0, len(pose_dict['num_standing']), 10)]
        pose_dict['num_sitting'] = [sum(pose_dict['num_sitting'][i:i+10])/10 for i in range(0, len(pose_dict['num_sitting']), 10)]

        sns.lineplot(data=pose_dict)

        # Save img
        plt.savefig(self.output_dir/'day.jpg')
        plt.cla()
        plt.clf()
        plt.close()