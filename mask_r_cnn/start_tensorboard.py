#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description='启动TensorBoard查看训练日志')
    parser.add_argument('--logdir', default='work_dirs', help='TensorBoard日志目录')
    parser.add_argument('--port', type=int, default=6006, help='TensorBoard服务器端口')
    return parser.parse_args()

def start_tensorboard(log_dir, port=6006):
    """启动TensorBoard服务器"""
    try:
        # 检查日志目录是否存在
        if not os.path.exists(log_dir):
            print(f'警告: 日志目录 {log_dir} 不存在，将创建该目录')
            os.makedirs(log_dir, exist_ok=True)
            
        # 检查是否已安装tensorboard
        subprocess.run(['tensorboard', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f'正在启动TensorBoard，日志目录: {log_dir}')
        # 启动TensorBoard服务器
        tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', log_dir, '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待TensorBoard启动
        time.sleep(3)
        
        print(f'TensorBoard已启动，请访问 http://localhost:{port} 查看训练过程')
        print('按Ctrl+C停止TensorBoard服务器...')
        
        # 保持程序运行，直到用户按下Ctrl+C
        while True:
            time.sleep(1)
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('未找到TensorBoard，请先安装: pip install tensorboard')
    except KeyboardInterrupt:
        print('\n正在关闭TensorBoard...')
        if 'tensorboard_process' in locals():
            tensorboard_process.terminate()
            tensorboard_process.wait()
        print('TensorBoard已关闭')

if __name__ == '__main__':
    args = parse_args()
    start_tensorboard(args.logdir, args.port)