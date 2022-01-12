# APiMoS
This is the official git repository for the Autonomous Pig Monitoring System (APiMoS).

## Getting Started

### Prerequisites
1. Install CUDA and CUDNN on your PC.

2. Create a conda environment with python 3.7 and activate it.
```
conda create -n apimos python=3.7 -y
conda activate apimos
```

3. Install latest version of [PyTorch](https://pytorch.org) and make sure that it fits your CUDA version. Example:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

4. Install MMCV, MMDetection and MMClassification.
```
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmcls
```

5. Other libraries used:
```
conda install -c conda-forge tqdm seaborn ffmpeg-python
```
___

### Setup APiMoS data
1. Download APiMoS data.

2. Place all data from `/all_data/configs/mmcls/` in the repo directory `./configs/mmcls/`. Same goes for mmdet configs.

3. Place `/all_data/data/seges_dataset_small/images` in repo directory `./data/seges_dataset_small/images`.

4. Place `/all_data/data/seges_dataset_small/dataset/cls_nm_c236789/latest.pth` in repo directory `./data/seges_dataset_small/dataset/cls_nm_c236789/`. Same goes for `det_ym_c236789`.

5. Place `inference_data` whereever you desire.

6. Duplicate one of the `setup_[name].config` in the repo `./configs/` and make your own version of it by changing paths to the desired locations.

----
### Run the System
To run inference. 
```
python ./run_system.py
```