from mmdet.apis import DetInferencer
inferencer = DetInferencer(
    ## for mask-rcnn
    weights='work_dirs/mask-rcnn_r50_fpn_1x_voc0712/epoch_12.pth',
    ## for sparse-rcnn
    # weights='work_dirs/sparse-rcnn_r50_fpn_1x_voc/epoch_20.pth',
    device='cuda:0'
)
## for mask-rcnn
inferencer('data/external_images', out_dir='outputs/mask', no_save_pred=False)

## for sparse-rcnn
#inferencer('demo/in', out_dir='outputs/in/sparse_cnn', no_save_pred=False)