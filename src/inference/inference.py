from tqdm import tqdm
from PIL import Image
import numpy as np
import asyncio
import time

from .src.detector import Detector
from .src.classifier import Classifier
from .src import utils


async def inference(config, workdir, streamlit_progress, timer):
    print('[INFO] Starting Inference')

    # Detector Model
    detector = Detector(streamqueue_size=5, timer=timer)
    detector.load_model(model_type=0, model_dir=config['detection']['det_dir'])

    # Classifier Model
    classifier = Classifier(streamqueue_size=5, timer=timer)
    classifier.load_model(model_type=1, model_dir=config['classification']['cls_dir'])

    for camera_path in utils.get_files(workdir):
        all_classifications = []
        all_files = utils.get_files(camera_path / 'images')

        for i, file_path in enumerate(tqdm(all_files)):

            # Update streamlit
            if streamlit_progress:
                streamlit_progress(i/len(all_files))

            # Load Images
            img = Image.open(file_path)
            img = np.asarray(img)

            # Inferene
            ## Detect Pig
            detections = await detector.detect(confidence=float(config['detection']['det_conf']), image=img)

            ## Classify Pig Pose
            if len(detections) > 0:
                classifications = await classifier.classify(img, detections, file_path, camera_path)
                all_classifications.append(classifications)

        utils.create_files(camera_path, all_classifications)
        return all_classifications


def main(config, work_dir, streamlit_progress=None, timer=None):
    all_classifications = asyncio.run(inference(config, work_dir, streamlit_progress, timer))
    return all_classifications


if __name__ == '__main__':
    main()
