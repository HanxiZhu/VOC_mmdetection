import os
import argparse
import json
from tensorboardX import SummaryWriter

def parse_log(log_path):
    train_loss = []
    val_map = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line)
            except Exception:
                continue
            # 训练loss
            if 'loss' in log and 'iter' in log and not any(k.startswith(('coco/', 'pascal_voc/', 'mAP')) for k in log):
                train_loss.append((log['iter'], log['loss']))

            # 验证集指标（VOC mAP 或 COCO mAP）
            for key in log:
                if key in ['pascal_voc/mAP', 'mAP', 'coco/bbox_mAP_50', 'coco/segm_mAP_50']:
                    val_map.append((log.get('step', log.get('epoch', 0)), log[key]))
                    break
    return train_loss, val_map


def log_to_tensorboard(log_path, tb_dir):
    train_loss, val_map = parse_log(log_path)
    print(f"train_loss: {len(train_loss)}, val_mAP: {len(val_map)}")
    writer = SummaryWriter(tb_dir)
    for step, loss in train_loss:
        writer.add_scalar('Train/Loss', loss, step)
    for step, map in val_map:
        writer.add_scalar('Val/mAP', map, step)
    writer.close()
    print(f"已写入TensorBoard日志到: {tb_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDet log.json to TensorBoard event file')
    parser.add_argument('log_json', help='Path to MMDetection log.json')
    parser.add_argument('--out', default='tf_logs', help='Output TensorBoard log dir')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_to_tensorboard(args.log_json, args.out)