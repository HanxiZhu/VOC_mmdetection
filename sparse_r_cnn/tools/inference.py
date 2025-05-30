from mmdet.apis import DetInferencer
inferencer = DetInferencer(
    weights='work_dirs/sparse-rcnn_r50_fpn_1x_voc/epoch_12.pth',
    device='cuda:0'
)

# inferencer('demo_sp/input', out_dir='demo_sp/outputs', no_save_pred=False)
inferencer('demo_sp/pics', out_dir='demo_sp/pics_outputs', no_save_pred=False)