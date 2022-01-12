from typing import NamedTuple
from .base_model import BaseModel
import mmcls.apis
from mmdet.utils.contextmanagers import concurrent
from ...position_estimation import inference
from PIL import Image
import time
from pathlib import Path

class Classification(NamedTuple):
    meta_data: dict
    coordinates: dict
    classification: dict


class Classifier(BaseModel):
    def __init__(self, streamqueue_size, timer):
        super().__init__(streamqueue_size, timer)

    async def classify(self, image, detections, file_path, camera_path):
        classifications = []
        
        for idx, bbox in enumerate(detections[0]):

            # Get Region of Interest
            x0, y0, x1, y1, det_conf = bbox
           # print('Original Image Shape: ', image.shape)
           # print('x0: ', x0, '  | y0:', y0, '  | x1: ', x1, '  | y1: ', y1)

            instance_img = image[int(y0):int(y1), int(x0):int(x1)]

           # print('RoI Shape: ', instance_img.shape)

            # Convert to world bbox
            est_start_time = time.time()
            world_bbox = inference.transform_bbox(bbox, camera_path.stem)
            self.timer.append('est', est_start_time)
            w_x, w_y = world_bbox[0][0][0], world_bbox[0][0][1]

            if 0 in instance_img.shape:
                print(instance_img.shape)
                continue


            # Start classifyig
            cls_start_time = time.time()
            async with concurrent(self.streamqueue):
                output = mmcls.apis.inference_model(self.model, instance_img)
                # cls = self.__setup_classification_info(output, idx, file_path, bbox=[x, y, w, h], det_conf=det_conf)
                cls = self.__setup_classification_info(output, idx, file_path, bbox=[x0, y0, x1, y1, w_x, w_y], det_conf=det_conf)
                
                classifications.append(cls)
            self.timer.append('cls', cls_start_time)

            # mmcls.apis.show_result_pyplot(self.model, instance_img, output)
        
        return classifications

    def __setup_classification_info(self, cls_info, idx, file_path, bbox, det_conf):
        cls_info['pred_label'] = str(cls_info['pred_label'])
        cls_info['pred_score'] = str(cls_info['pred_score'])

        coordinates = {'x0': str(bbox[0]), 'y0': str(bbox[1]), 'x1': str(bbox[2]), 'y1': str(bbox[3]), 'world_x': str(bbox[4]), 'world_y': str(bbox[5])}

        file_path = Path(*file_path.parts[-3:])

        meta_data = {'detection_image': str(file_path), 'roi_image': f'roi_{idx}', 'detection_confidence': str(det_conf)}
        cls = {**meta_data, **coordinates, **cls_info}

        # cls = Classification(meta_data=meta_data, coordinates=coordinates, classification=cls_info)

        return cls
