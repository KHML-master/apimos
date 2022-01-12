from .base_model import BaseModel
import mmdet.apis
from mmdet.utils.contextmanagers import concurrent
import numpy as np
import time


class Detector(BaseModel):
    def __init__(self, streamqueue_size, timer):
        super().__init__(streamqueue_size, timer)

    async def detect(self, confidence, image):

        start_time = time.time()

        # Start detecting queue
        async with concurrent(self.streamqueue):
            detections = await mmdet.apis.async_inference_detector(self.model, image)

        # Confidence Threshold
        detections = self.__confidence_threshold(threshold=confidence, detections=detections)

        self.timer.append('det', start_time)

        # Return result
        return detections

    def __confidence_threshold(self, threshold, detections):
        detections = np.array([[d for d in detections[0] if d[4] > threshold]])
        return detections
