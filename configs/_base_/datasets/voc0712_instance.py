custom_imports = dict(
    imports=['custom_datasets'],
    allow_failed_imports=False
)

# VOC 类别定义
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 训练流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')

]

# 测试流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')

]

# 数据根目录
data_root = 'data/coco/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'

# 数据集类型
dataset_type = 'VOCInstanceDataset'


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Segmentation/train.txt',
        data_prefix=dict(
            img_path='JPEGImages/',
            ann_path='Annotations/',
            seg_map_path='SegmentationObject/'
        ),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages/',
            ann_path='Annotations/',
            seg_map_path='SegmentationObject/'
        ),
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes)
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

# 测试数据加载器（复用验证集）
test_dataloader = val_dataloader

# 验证评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=None,
    metric=['bbox', 'segm']
)



# 测试评估器
test_evaluator = val_evaluator
