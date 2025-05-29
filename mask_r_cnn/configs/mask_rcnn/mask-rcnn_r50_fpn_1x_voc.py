#from custom_datasets import VOCDatasetInstance
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    #'../_base_/datasets/voc0712_instance.py',
    '../_base_/models/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# train_cfg = dict(max_epochs=100, val_interval=7)
# device = 'cuda'
# dataset_type = 'VOCDataset'
# data_root = 'data/coco/VOCtrainval_11-May-2012/VOCdevkit/'
# classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# # 使用 VOC2007 + VOC2012 的 trainval.txt 联合训练
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=20),
#         mask_head=dict(num_classes=20)))
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2012/ImageSets/Main/train.txt',
#         img_prefix=data_root + 'VOC2012/',
#         classes=classes
#     ),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2012/ImageSets/Main/trainval.txt',
#         img_prefix=data_root + 'VOC2012/',
#         classes=classes
#     ),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2012/ImageSets/Main/train_val.txt',
#         img_prefix=data_root + 'VOC2012/',
#         classes=classes
#     )
# )

# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

