import os
import mmcv
import numpy as np
import cv2
import torch
from mmdet.apis import init_detector, inference_detector

config_file = 'mmdetection/configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py'
checkpoint_file = 'work_dirs/mask-rcnn_r50_fpn_1x_voc0712/epoch_12.pth'
img_dir = 'data/external_images'
output_dir = 'outputs/vis'
os.makedirs(output_dir, exist_ok=True)

model = init_detector(config_file, checkpoint_file, device='cuda:0')
class_names = model.dataset_meta['classes']
img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_path in img_list:
    img = mmcv.imread(img_path)
    # --------- 1. 可视化 proposal box（RPN候选框） ---------
    with torch.no_grad():
        device = next(model.parameters()).device
        # 预处理图片为tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        # 归一化（根据你的config文件设置，若有mean/std请手动归一化）
        img_tensor = img_tensor / 255.0
        # 提取特征
        feats = model.extract_feat(img_tensor)
        # 获取proposal（MMDet 3.x RPNHead没有simple_test，直接用forward）
        rpn_outs = model.rpn_head(feats)
        proposal_cfg = model.test_cfg.rpn
        proposal_list = model.rpn_head.predict_by_feat(
            *rpn_outs,
            batch_img_metas=[{'img_shape': img.shape, 'scale_factor': 1.0, 'pad_shape': img.shape, 'ori_shape': img.shape}],
            cfg=proposal_cfg
        )[0].cpu().numpy()
    # 绘制前100个proposal
    proposal_img = img.copy()
    # proposal_list 可能是 InstanceData，需要取 .bboxes
    if hasattr(proposal_list, 'bboxes'):
        proposal_boxes = proposal_list.bboxes
        if hasattr(proposal_boxes, 'cpu'):
            proposal_boxes = proposal_boxes.cpu().numpy()
        else:
            proposal_boxes = np.array(proposal_boxes)
    else:
        proposal_boxes = proposal_list  # 已经是 numpy

    for box in proposal_boxes[:100]:
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(proposal_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    mmcv.imwrite(proposal_img, os.path.join(output_dir, os.path.basename(img_path).replace('.', '_proposal.')))

    # --------- 2. 可视化最终预测结果（检测框/分割） ---------
    result = inference_detector(model, img_path)
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, 'masks') and pred_instances.masks is not None else None

    vis_img = img.copy()
    color_map = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for idx, bbox in enumerate(bboxes):
        if scores[idx] < 0.3:
            continue
        color = color_map[labels[idx] % len(color_map)]
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label_text = f'{class_names[labels[idx]]}: {scores[idx]:.2f}'
        cv2.putText(vis_img, label_text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if masks is not None:
            mask = masks[idx].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, 2)
            vis_img[mask > 0] = vis_img[mask > 0] * 0.5 + np.array(color) * 0.5

    mmcv.imwrite(vis_img, os.path.join(output_dir, os.path.basename(img_path).replace('.', '_final.')))

print(f'proposal和最终预测结果已保存到 {output_dir}')