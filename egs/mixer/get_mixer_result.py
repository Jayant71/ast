# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Generated for custom mixer dataset
# @File    : get_mixer_result.py

# Summarize mixer dataset training result

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")

if __name__ == '__main__':
    args = parser.parse_args()

    result_file = os.path.join(args.exp_path, 'result.csv')
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        exit(1)

    result = np.loadtxt(result_file, delimiter=',')

    num_cols = result.shape[1] if result.ndim > 1 else 1
    print('--------------Mixer Dataset Result Summary--------------')
    if num_cols >= 10:
        print('Epoch | Accuracy | AUC | Precision | Recall | d_prime | Train Loss | Val Loss | Cum_Acc | Cum_AUC | LR')
    else:
        print('Epoch | Accuracy | AUC | Precision | Recall | d_prime | Train Loss | Val Loss | LR')
    print('-' * 110)

    best_epoch = np.argmax(result[:, 0])
    best_result = result[best_epoch, :]

    for epoch in range(len(result)):
        epoch_result = result[epoch]
        if num_cols >= 10:
            print(f'{epoch+1:5d} | {epoch_result[0]:8.4f} | {epoch_result[1]:6.4f} | {epoch_result[2]:9.4f} | {epoch_result[3]:6.4f} | {epoch_result[4]:7.4f} | {epoch_result[5]:10.4f} | {epoch_result[6]:8.4f} | {epoch_result[7]:7.4f} | {epoch_result[8]:7.4f} | {epoch_result[9]:6.4f}')
        else:
            print(f'{epoch+1:5d} | {epoch_result[0]:8.4f} | {epoch_result[1]:6.4f} | {epoch_result[2]:9.4f} | {epoch_result[3]:6.4f} | {epoch_result[4]:7.4f} | {epoch_result[5]:10.4f} | {epoch_result[6]:8.4f} | {epoch_result[7]:6.4f}')

    print('-' * 110)
    print(f'Best epoch: {best_epoch + 1}')
    print(f'Best accuracy: {best_result[0]:.4f}')
    print(f'Best AUC: {best_result[1]:.4f}')

    np.savetxt(os.path.join(args.exp_path, 'best_result.csv'), best_result, delimiter=',')

    print(f'Best result saved to {args.exp_path}/best_result.csv')