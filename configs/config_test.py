# configs/config_test.py

# 继承 pascal_voc 下的 Mask R-CNN on VOC0712 全配置
_base_ = [
    '../mmdetection/configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
]

# -------------------------------------------------------------------
# 只修改测试集的路径：
# - ann_file 指向 data/VOCdevkit/VOC2007/ImageSets/Main/test.txt
# - img_prefix 指向 data/VOCdevkit/VOC2007/JPEGImages/
# pipeline 使用基配置里定义好的 test_pipeline
# -------------------------------------------------------------------
data = dict(
    test=dict(
        type='VOCInstanceDataset',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007/JPEGImages/',
    )
)

# 同时评估检测和分割
evaluation = dict(metric=['bbox', 'segm'])
