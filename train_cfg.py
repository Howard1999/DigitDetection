_base_ = 'mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)
    ),
)

dataset_type = 'CocoDataset'
classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')
data = dict(
    train=dict(
        img_prefix='../data/train/',
        classes=classes,
        ann_file='../data/train.json'),
    val=dict(
        img_prefix='../data/val/',
        classes=classes,
        ann_file='../data/val.json'),
    test=dict(
        img_prefix='../data/test/',
        classes=classes,
        ann_file='../data/test.json')
)

optimizer = dict(lr=2e-3)
