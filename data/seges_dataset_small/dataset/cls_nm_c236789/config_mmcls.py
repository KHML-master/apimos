# Based
# https://github.com/open-mmlab/mim-example/tree/master/nuimages_seg
# Command: PYTHONPATH='.'$PYTHONPATH mim train mmcls ./config_mmcls.py --work-dir ./tmp/ --gpus 1       
_base_ = ['../../../../configs/mmcls/resnet/resnet18_b32x8_imagenet.py']

dataset_type = 'PigVision'
custom_imports = dict(imports=['pig_vision'], allow_failed_imports=False)


# Data Settings
base_dir = '.'
data_dir = f'{base_dir}/../../images'
classes = ('standing', 'sternal', 'lateral', 'sitting',)


data = dict(
    train=dict(
        type='PigVision',
        classes=classes,
        ann_file=f'{base_dir}/train.json',
        data_prefix=data_dir
    ),
    val=dict(
        type='PigVision',
        classes=classes,
        ann_file=f'{base_dir}/valid.json',
        data_prefix=data_dir
    ),
    test=dict(
        type='PigVision',
        classes=classes,
        ann_file=f'{base_dir}/test.json',
        data_prefix=data_dir
    )
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224, keep_ratio=True),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='rotate', angle=10),
    dict(type='cutout', shape=[30.0,30.0]),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_A_train = dict(
    dataset=dict(
        type='PigVision',
        ann_file = f'./cls_dataset.json',
        pipeline=train_pipeline
    )
)

model = dict(
    head = dict(
        num_classes=4,
        topk=(1, 4),
    )
)