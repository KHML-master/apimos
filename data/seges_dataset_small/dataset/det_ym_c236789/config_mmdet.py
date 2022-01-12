# Based
_base_ = '../../../../configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py'

# Data Settings
base_dir = '.'
data_dir = f'{base_dir}/../../images'
classes = ('pig',)

data = dict(
    train=dict(
        classes=classes,
        ann_file=f'{base_dir}/train.json',
        img_prefix=data_dir
    ),
    val=dict(
        classes=classes,
        ann_file=f'{base_dir}/validation.json',
        img_prefix=data_dir
    ),
    test=dict(
        classes=classes,
        ann_file=f'{base_dir}/test.json',
        img_prefix=data_dir
    )
)

# Model ResNet
model = dict(
    backbone=dict(
        type='ResNet',
        with_cp=True
    ),
    roi_head=dict(
        bbox_head=dict(
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1
            )
        )
    )
)
