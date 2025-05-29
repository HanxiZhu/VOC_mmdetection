import os
import argparse
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import random
import subprocess
import time
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

# 注册所有模块
register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='Train and test models on VOC dataset')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--test', action='store_true', help='Test the models')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    parser.add_argument('--external', action='store_true', help='Visualize results on external images')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard visualization')
    parser.add_argument('--tensorboard-port', type=int, default=6006, help='Port for TensorBoard server')
    parser.add_argument('--model', choices=['mask-rcnn', 'sparse-rcnn', 'both'], default='mask-rcnn', help='选择要训练的模型: mask-rcnn, sparse-rcnn 或 both')
    parser.add_argument('--mask-rcnn-config', default='mmdetection/configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py', help='Config file for Mask R-CNN')
    parser.add_argument('--sparse-rcnn-config', default='mmdetection/configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py', help='Config file for Sparse R-CNN')
    parser.add_argument('--mask-rcnn-checkpoint', default='work_dirs/mask-rcnn_r50_fpn_1x_voc0712/epoch_12.pth', help='Checkpoint file for Mask R-CNN')
    parser.add_argument('--sparse-rcnn-checkpoint', default='work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/epoch_12.pth', help='Checkpoint file for Sparse R-CNN')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def train_model(config_file):
    # 加载配置文件
    cfg = Config.fromfile(config_file)
    
    # 修改数据集路径
    cfg.data_root = 'data/VOCdevkit/'
    
    # 设置工作目录
    model_name = os.path.splitext(os.path.basename(config_file))[0]
    cfg.work_dir = f'work_dirs/{model_name}'
    
    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 启用TensorBoard可视化
    if args.tensorboard:
        # 确保visualizer配置存在
        if not hasattr(cfg, 'visualizer'):
            cfg.visualizer = dict(
                type='DetLocalVisualizer',
                vis_backends=[
                    dict(type='LocalVisBackend'),
                ],
                name='visualizer'
            )
        # 添加TensorBoard后端
        if not hasattr(cfg.visualizer, 'vis_backends'):
            cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]
        
        # 检查是否已经有TensorboardVisBackend
        has_tensorboard_backend = False
        for backend in cfg.visualizer.vis_backends:
            if backend.get('type') == 'TensorboardVisBackend':
                has_tensorboard_backend = True
                break
        
        # 如果没有，添加TensorboardVisBackend
        if not has_tensorboard_backend:
            cfg.visualizer.vis_backends.append(dict(type='TensorboardVisBackend'))
        
        print(f'已启用TensorBoard可视化，日志将保存在 {cfg.work_dir} 目录下')
    
    # 创建训练Runner
    runner = Runner.from_cfg(cfg)
    
    # 开始训练
    runner.train()

def test_model(config_file, checkpoint_file):
    # 加载配置文件
    cfg = Config.fromfile(config_file)
    
    # 修改数据集路径
    cfg.data_root = 'data/VOCdevkit/'
    
    # 设置工作目录
    model_name = os.path.splitext(os.path.basename(config_file))[0]
    cfg.work_dir = f'work_dirs/{model_name}'
    
    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 启用TensorBoard可视化
    if args.tensorboard:
        # 确保visualizer配置存在
        if not hasattr(cfg, 'visualizer'):
            cfg.visualizer = dict(
                type='DetLocalVisualizer',
                vis_backends=[
                    dict(type='LocalVisBackend'),
                ],
                name='visualizer'
            )
        # 添加TensorBoard后端
        if not hasattr(cfg.visualizer, 'vis_backends'):
            cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]
        
        # 检查是否已经有TensorboardVisBackend
        has_tensorboard_backend = False
        for backend in cfg.visualizer.vis_backends:
            if backend.get('type') == 'TensorboardVisBackend':
                has_tensorboard_backend = True
                break
        
        # 如果没有，添加TensorboardVisBackend
        if not has_tensorboard_backend:
            cfg.visualizer.vis_backends.append(dict(type='TensorboardVisBackend'))
        
        print(f'已启用TensorBoard可视化，测试结果将保存在 {cfg.work_dir} 目录下')
    
    # 加载模型
    model = init_detector(config_file, checkpoint_file, device=args.device)
    
    # 创建测试Runner
    runner = Runner.from_cfg(cfg)
    
    # 开始测试
    runner.test()
    
    return model

def visualize_proposal_vs_final(model, test_img_paths, save_dir='results/proposal_vs_final'):
    import torchvision.transforms.functional as F  # 若未导入可能需补充

    os.makedirs(save_dir, exist_ok=True)
    visualizer = DetLocalVisualizer()

    for i, img_path in enumerate(test_img_paths):
        # 原始图像
        img = mmcv.imread(img_path)
        img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')

        # 转为 tensor 并送入模型
        img_tensor = torch.from_numpy(img_rgb).unsqueeze(0).permute(0, 3, 1, 2).float().to(args.device)

        # 特征提取
        feats = model.extract_feat(img_tensor)

        # RPN forward 返回 cls_scores 和 bbox_preds
        cls_scores, bbox_preds = model.rpn_head.forward(feats)

        # 计算 proposal boxes（注意 img_metas 要手动构造）
        img_metas = [{'img_shape': img.shape[:2]}]
        proposals = model.rpn_head.predict_by_feat(cls_scores, bbox_preds, img_metas)[0]

        # 最终检测结果
        result = inference_detector(model, img_path)
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(img_rgb)
        axes[0].set_title('Proposal Boxes')
        axes[1].imshow(img_rgb)
        axes[1].set_title('Final Detection Results')

        # 绘制 proposal boxes（蓝色）
        for box in proposals.cpu().numpy():
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, edgecolor='blue', linewidth=1)
            axes[0].add_patch(rect)

        # 绘制最终检测结果（红色）
        for box, score, label in zip(bboxes, scores, labels):
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(box[0], box[1], f'{model.dataset_meta["classes"][label]}: {score:.2f}',
                         bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{i + 1}.png'))
        plt.close()

def visualize_comparison(mask_rcnn_model, sparse_rcnn_model, test_img_paths, save_dir='results/model_comparison'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    for i, img_path in enumerate(test_img_paths):
        # 推理
        mask_rcnn_result = inference_detector(mask_rcnn_model, img_path)
        sparse_rcnn_result = inference_detector(sparse_rcnn_model, img_path)
        
        # 获取原始图像
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # 显示原始图像
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 1].imshow(img)
        axes[0, 1].set_title('Mask R-CNN Detection')
        axes[1, 0].imshow(img)
        axes[1, 0].set_title('Mask R-CNN Segmentation')
        axes[1, 1].imshow(img)
        axes[1, 1].set_title('Sparse R-CNN Segmentation')
        
        # 绘制Mask R-CNN检测结果
        mask_bboxes = mask_rcnn_result.pred_instances.bboxes.cpu().numpy()
        mask_scores = mask_rcnn_result.pred_instances.scores.cpu().numpy()
        mask_labels = mask_rcnn_result.pred_instances.labels.cpu().numpy()
        mask_masks = mask_rcnn_result.pred_instances.masks.cpu().numpy() if hasattr(mask_rcnn_result.pred_instances, 'masks') else None
        
        for box, score, label in zip(mask_bboxes, mask_scores, mask_labels):
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                            fill=False, edgecolor='red', linewidth=2)
            axes[0, 1].add_patch(rect)
            axes[0, 1].text(box[0], box[1], f'{mask_rcnn_model.dataset_meta["classes"][label]}: {score:.2f}',
                        bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
        
        # 绘制Mask R-CNN分割结果
        if mask_masks is not None:
            colors = [plt.cm.get_cmap('hsv', 21)(i) for i in range(21)]
            for mask, label in zip(mask_masks, mask_labels):
                color = colors[label]
                mask_img = np.zeros_like(img)
                mask_img[mask] = [c * 255 for c in color[:3]]
                axes[1, 0].imshow(mask_img, alpha=0.5)
        
        # 绘制Sparse R-CNN分割结果
        sparse_bboxes = sparse_rcnn_result.pred_instances.bboxes.cpu().numpy()
        sparse_scores = sparse_rcnn_result.pred_instances.scores.cpu().numpy()
        sparse_labels = sparse_rcnn_result.pred_instances.labels.cpu().numpy()
        sparse_masks = sparse_rcnn_result.pred_instances.masks.cpu().numpy() if hasattr(sparse_rcnn_result.pred_instances, 'masks') else None
        
        for box, score, label in zip(sparse_bboxes, sparse_scores, sparse_labels):
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                            fill=False, edgecolor='blue', linewidth=2)
            axes[1, 1].add_patch(rect)
            axes[1, 1].text(box[0], box[1], f'{sparse_rcnn_model.dataset_meta["classes"][label]}: {score:.2f}',
                        bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')
        
        if sparse_masks is not None:
            colors = [plt.cm.get_cmap('hsv', 21)(i) for i in range(21)]
            for mask, label in zip(sparse_masks, sparse_labels):
                color = colors[label]
                mask_img = np.zeros_like(img)
                mask_img[mask] = [c * 255 for c in color[:3]]
                axes[1, 1].imshow(mask_img, alpha=0.5)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{i+1}.png'))
        plt.close()

def visualize_external_images(mask_rcnn_model, sparse_rcnn_model, external_img_dir='Data/external_images', save_dir='results/external_images'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取外部图像路径
    external_img_paths = [os.path.join(external_img_dir, f) for f in os.listdir(external_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for i, img_path in enumerate(external_img_paths):
        # 推理
        mask_rcnn_result = inference_detector(mask_rcnn_model, img_path)
        sparse_rcnn_result = inference_detector(sparse_rcnn_model, img_path)
        
        # 获取原始图像
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # 显示原始图像
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[1].imshow(img)
        axes[1].set_title('Mask R-CNN Result')
        axes[2].imshow(img)
        axes[2].set_title('Sparse R-CNN Result')
        
        # 绘制Mask R-CNN结果
        mask_bboxes = mask_rcnn_result.pred_instances.bboxes.cpu().numpy()
        mask_scores = mask_rcnn_result.pred_instances.scores.cpu().numpy()
        mask_labels = mask_rcnn_result.pred_instances.labels.cpu().numpy()
        mask_masks = mask_rcnn_result.pred_instances.masks.cpu().numpy() if hasattr(mask_rcnn_result.pred_instances, 'masks') else None
        
        for box, score, label in zip(mask_bboxes, mask_scores, mask_labels):
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                            fill=False, edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(box[0], box[1], f'{mask_rcnn_model.dataset_meta["classes"][label]}: {score:.2f}',
                        bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
        
        if mask_masks is not None:
            colors = [plt.cm.get_cmap('hsv', 21)(i) for i in range(21)]
            for mask, label in zip(mask_masks, mask_labels):
                color = colors[label]
                mask_img = np.zeros_like(img)
                mask_img[mask] = [c * 255 for c in color[:3]]
                axes[1].imshow(mask_img, alpha=0.5)
        
        # 绘制Sparse R-CNN结果
        sparse_bboxes = sparse_rcnn_result.pred_instances.bboxes.cpu().numpy()
        sparse_scores = sparse_rcnn_result.pred_instances.scores.cpu().numpy()
        sparse_labels = sparse_rcnn_result.pred_instances.labels.cpu().numpy()
        sparse_masks = sparse_rcnn_result.pred_instances.masks.cpu().numpy() if hasattr(sparse_rcnn_result.pred_instances, 'masks') else None
        
        for box, score, label in zip(sparse_bboxes, sparse_scores, sparse_labels):
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                            fill=False, edgecolor='blue', linewidth=2)
            axes[2].add_patch(rect)
            axes[2].text(box[0], box[1], f'{sparse_rcnn_model.dataset_meta["classes"][label]}: {score:.2f}',
                        bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')
        
        if sparse_masks is not None:
            colors = [plt.cm.get_cmap('hsv', 21)(i) for i in range(21)]
            for mask, label in zip(sparse_masks, sparse_labels):
                color = colors[label]
                mask_img = np.zeros_like(img)
                mask_img[mask] = [c * 255 for c in color[:3]]
                axes[2].imshow(mask_img, alpha=0.5)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'external_{i+1}.png'))
        plt.close()

def get_test_images(num_images=4):
    # 获取测试集图像路径
    test_img_dir = 'data/VOCdevkit/VOC2007/JPEGImages'
    test_img_list = 'data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    
    with open(test_img_list, 'r') as f:
        img_ids = [line.strip() for line in f if line.strip()]
    
    # 随机选择图像
    selected_ids = random.sample(img_ids, num_images)
    test_img_paths = [os.path.join(test_img_dir, f'{img_id}.jpg') for img_id in selected_ids]
    
    return test_img_paths

def start_tensorboard(log_dir, port=6006):
    """启动TensorBoard服务器"""
    try:
        # 检查是否已安装tensorboard
        subprocess.run(['tensorboard', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 启动TensorBoard服务器
        tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', log_dir, '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待TensorBoard启动
        time.sleep(3)
        
        print(f'TensorBoard已启动，请访问 http://localhost:{port} 查看训练过程')
        return tensorboard_process
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('未找到TensorBoard，请先安装: pip install tensorboard')
        return None

if __name__ == '__main__':
    args = parse_args()
    
    # 启动TensorBoard
    tensorboard_process = None
    if args.tensorboard:
        log_dir = 'work_dirs'
        tensorboard_process = start_tensorboard(log_dir, args.tensorboard_port)
    
    # 训练模型
    if args.train:
        if args.model in ['mask-rcnn', 'both']:
            print('Training Mask R-CNN...')
            train_model(args.mask_rcnn_config)
        
        if args.model in ['sparse-rcnn', 'both']:
            print('Training Sparse R-CNN...')
            train_model(args.sparse_rcnn_config)
    
    # 测试模型
    if args.test or args.visualize or args.external:
        print('Loading Mask R-CNN model...')
        mask_rcnn_model = init_detector(args.mask_rcnn_config, args.mask_rcnn_checkpoint, device=args.device)
        
        print('Loading Sparse R-CNN model...')
        sparse_rcnn_model = init_detector(args.sparse_rcnn_config, args.sparse_rcnn_checkpoint, device=args.device)
        
        if args.test:
            print('Testing Mask R-CNN...')
            test_model(args.mask_rcnn_config, args.mask_rcnn_checkpoint)
            
            print('Testing Sparse R-CNN...')
            test_model(args.sparse_rcnn_config, args.sparse_rcnn_checkpoint)
    
    # 可视化结果
    if args.visualize:
        # 获取测试图像
        test_img_paths = get_test_images(4)
        
        print('Visualizing proposal boxes vs final detection results for Mask R-CNN...')
        visualize_proposal_vs_final(mask_rcnn_model, test_img_paths)
        
        print('Visualizing comparison between Mask R-CNN and Sparse R-CNN...')
        visualize_comparison(mask_rcnn_model, sparse_rcnn_model, test_img_paths)
    
    # 可视化外部图像结果
    if args.external:
        print('Visualizing results on external images...')
        visualize_external_images(mask_rcnn_model, sparse_rcnn_model)
    
    # 如果TensorBoard正在运行，等待用户手动关闭
    if tensorboard_process is not None:
        print('\nTensorBoard正在运行，按Ctrl+C停止程序...')
        try:
            # 保持程序运行，直到用户按下Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('正在关闭TensorBoard...')
            tensorboard_process.terminate()
            tensorboard_process.wait()
        
    print('Done!')