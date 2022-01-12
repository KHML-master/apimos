from src.inference import inference
from src.data_preprocessing.video_to_frames import video_to_frames
from src.visualisations.visualise_inference import visualise_inference as visualise
import os
from pathlib import Path
import time
import configparser
import csv

class Timer:
    def __init__(self):
        self.cls_times = []
        self.det_times = []
        self.est_times = []

    def append(self, module_type, module_time):
        duration = time.time()-module_time
        if module_type == 'det':
            self.det_times.append(duration)
        elif module_type == 'cls':
            self.cls_times.append(duration)
        elif module_type == 'est':
            self.est_times.append(duration)

    def build_csv(self, work_dir):
        with open(f'{work_dir}/time_info.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([self.cls_times, self.det_times, self.est_times])



if __name__ == '__main__':
    timer = Timer()

    # Setup work directory
    work_dir = Path(f'./work_dir/{time.strftime("%Y_%d%m_%H%M%S")}')
    os.mkdir(work_dir)

    # Setup Config
    config = configparser.ConfigParser()
    config.read('./configs/setup_kasperpc.config')

    # Videos to frames (Input: Videos | Output: Images)
    cameras = video_to_frames.setup_cameras(config['pre_process']['input_data_path'], False, work_dir)
    for camera in cameras:
        camera.videos_to_frames(config['pre_process']['mask_path'])  # Convert videos into written frames
    print('Completed pre processing!')

    # Inference (Input: Images | Output: JSON)
    all_classifications = inference.main(config, work_dir, timer=timer)
    timer.build_csv(work_dir)

    # Visualize 
    visualise.inference(config, 'cam005', work_dir)
