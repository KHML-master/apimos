import mmcv

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.base_dataset import BaseDataset

import numpy as np
import json

@DATASETS.register_module()
class PigVision(BaseDataset):

    CLASSES = ('stomach', 'standing', 'side', 'sitting')

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            json_data = json.load(f)
            for img in json_data['images']:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': img['file_name']}
                info['gt_label'] = np.array(int(img['category_id']), dtype=np.int64)
                data_infos.append(info)
            return data_infos