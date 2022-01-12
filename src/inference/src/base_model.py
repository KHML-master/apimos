import asyncio
import torch
import mmdet.apis
import mmcls.apis


class BaseModel:

    def __init__(self, streamqueue_size, timer):
        # DNN Model
        self.model = None

        self.timer = timer

        self.device = 'cuda:0'
        # self.device = 'cpu'

        # Setting up Streamqueue
        self.streamqueue = self.__setup_streamqueue(streamqueue_size)

    def load_model(self, model_type, model_dir):
        # Build Model
        if model_type == 0:
            self.model = mmdet.apis.init_detector(config=f'{model_dir}/config_mmdet.py', checkpoint=f'{model_dir}/latest.pth', device=self.device)
        if model_type == 1:
            self.model = mmcls.apis.init_model(config=f'{model_dir}/config_mmcls.py', checkpoint=f'{model_dir}/latest.pth', device=self.device)

    def __setup_streamqueue(self, streamqueue_size):
        streamqueue = asyncio.Queue()

        # Queue size defines concurrency level
        streamqueue_size = streamqueue_size

        for _ in range(streamqueue_size):
            streamqueue.put_nowait(torch.cuda.Stream(device=self.device))

        return streamqueue
