# VOC 目标检测与实例分割项目（Mask R-CNN & Sparse R-CNN）

本项目基于 MMDetection 框架，分别实现了 **Mask R-CNN** 和 **Sparse R-CNN** 两种主流模型在 PASCAL VOC 数据集上的训练、测试、推理与可视化，支持 TensorBoard 实时监控训练过程。

---

## 📁 项目结构

```
.
├── mask_r_cnn/
│   ├── configs/
│   ├── data/external_images/
│   ├── mmdetection/configs/
│   ├── outputs/vis/
│   ├── work_dirs/
│   ├── inference.py             # 外部图像推理
│   ├── start_tensorboard.py     # 启动 TensorBoard
│   ├── test.py                  # 模型测试
│   ├── train_test_models.py     # 模型训练主脚本
│   └── visualize.py             # 可视化检测结果
│
├── sparse_r_cnn/
│   ├── configs/
│   ├── demo_sp/
│   ├── tools/
│   ├── work_dirs/sparse-rcnn_r50_fpn_1x_voc/
│   └── inference.py             # 外部图像推理
```

---

## 📦 环境配置

```bash
# 安装 PyTorch（请根据 CUDA 版本选择）
pip install torch torchvision

# 安装 MMDetection 相关依赖
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet

# 其他依赖
pip install matplotlib numpy tensorboard
```

---

## 📂 数据集准备

项目使用 PASCAL VOC 2007 和 2012 数据集，请解压数据至以下目录：

```
mask_r_cnn/data/VOCdevkit/
├── VOC2007/
├── VOC2012/
```

外部测试图像请放在：

```
mask_r_cnn/data/external_images/
```

---

## 🏋️‍♂️ 模型训练

### Mask R-CNN

```bash
cd mask_r_cnn
python train_test_models.py --train
```

### Sparse R-CNN

```bash
cd sparse_r_cnn
python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py
```

---

## ✅ 模型测试

### Mask R-CNN

```bash
cd mask_r_cnn
python test.py
```

### Sparse R-CNN

```bash
cd sparse_r_cnn
python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py     --checkpoint work_dirs/sparse-rcnn_r50_fpn_1x_voc/epoch_12.pth
```

---

## 🖼️ 模型推理与可视化

### 推理（外部图像）

```bash
# Mask R-CNN
cd mask_r_cnn
python inference.py

# Sparse R-CNN
cd sparse_r_cnn
python inference.py
```

### 可视化（训练集或测试集检测结果）

```bash
cd mask_r_cnn
python visualize.py
```

---

## 📊 TensorBoard 可视化训练过程

```bash
cd mask_r_cnn
python start_tensorboard.py
```

或手动运行：

```bash
tensorboard --logdir work_dirs
```

然后在浏览器中访问：http://localhost:6006

---

## 🧠 模型简介

| 模型        | 骨干网络       | 特征融合 | Proposal机制     | 实例分割头 | 推理方式    |
|-------------|----------------|----------|------------------|------------|-------------|
| Mask R-CNN  | ResNet-50 + FPN| FPN      | RPN + RoIAlign   | 有         | Two-stage   |
| Sparse R-CNN| ResNet-50 + FPN| FPN      | 无（query学习）  | 有         | End-to-End  |

---

## 📚 参考资料

- [MMDetection 文档](https://mmdetection.readthedocs.io/)
- [Mask R-CNN 论文](https://arxiv.org/abs/1703.06870)
- [Sparse R-CNN 论文](https://arxiv.org/abs/2011.12450)
- [PASCAL VOC 数据集官网](http://host.robots.ox.ac.uk/pascal/VOC/)

---

## 📝 License

本项目采用 MIT 协议，详见 LICENSE 文件。
