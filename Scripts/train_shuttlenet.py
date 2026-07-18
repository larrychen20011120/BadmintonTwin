#!/usr/bin/env python3
"""
ShuttleNet training script.

Usage:
    python Scripts/train_shuttlenet.py --experiment norm
    python Scripts/train_shuttlenet.py --experiment shifted
    python Scripts/train_shuttlenet.py --experiment norm --epochs 20 --batch-size 128
"""

import argparse
import os
import pickle
import subprocess
import sys

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from Utils.runtime_report import algorithm_runtime_rows, print_runtime_box


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_experiment_paths(config, experiment_type, data_dir=None, output_path=None):
    if experiment_type not in ('norm', 'shifted'):
        raise ValueError(f'Unsupported experiment type: {experiment_type}')

    split_path = config['paths'][f'{experiment_type}_split']
    data_dir = data_dir or config['paths']['shuttlenet']['data'][experiment_type]
    output_path = output_path or config['paths']['shuttlenet']['models'][experiment_type]
    return split_path, data_dir, output_path


def train_shuttlenet(experiment_type, epochs=None, batch_size=None, lr=None,
                     output_name=None, data_dir=None, encode_length=None,
                     max_ball_round=None, shot_dim=None, area_dim=None,
                     player_dim=None, encode_dim=None, area_num=None,
                     dropout=None, device=None, reprepare=False):
    config = load_config()
    train_config = config['shuttlenet_training']

    output_path = None
    if output_name:
        output_path = os.path.join(config['paths']['shuttlenet']['weight_dir'], output_name)

    split_path, data_dir, output_path = resolve_experiment_paths(
        config,
        experiment_type,
        data_dir=data_dir,
        output_path=output_path,
    )

    epochs = epochs if epochs is not None else train_config['epochs']
    batch_size = batch_size if batch_size is not None else train_config['batch_size']
    lr = lr if lr is not None else train_config['learning_rate']
    encode_length = encode_length if encode_length is not None else train_config['encode_length']
    max_ball_round = max_ball_round if max_ball_round is not None else train_config['max_ball_round']
    shot_dim = shot_dim if shot_dim is not None else train_config['shot_dim']
    area_dim = area_dim if area_dim is not None else train_config['area_dim']
    player_dim = player_dim if player_dim is not None else train_config['player_dim']
    encode_dim = encode_dim if encode_dim is not None else train_config['encode_dim']
    area_num = area_num if area_num is not None else train_config['area_num']
    dropout = dropout if dropout is not None else train_config['dropout']
    device = device or config['evaluation'].get('device', 'cuda')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print_runtime_box(
        'Runtime Parameters: train_shuttlenet',
        algorithm_runtime_rows(
            config,
            'shuttlenet',
            experiment=experiment_type,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            encode_length=encode_length,
            max_ball_round=max_ball_round,
        ) + [
            ('split_path', split_path),
            ('data_dir', data_dir),
            ('output_path', output_path),
            ('reprepare', reprepare),
        ],
    )

    print('=' * 70)
    print(f'ShuttleNet 模型訓練 - {experiment_type.upper()}')
    print('=' * 70)
    print(f'資料來源: {data_dir}')
    print(f'輸出模型: {output_path}')
    print('訓練參數:')
    print(f'  - Epochs:         {epochs}')
    print(f'  - Batch Size:     {batch_size}')
    print(f'  - Learning Rate:  {lr}')
    print(f'  - Encode Length:  {encode_length}')
    print(f'  - Max Ball Round: {max_ball_round}')
    print('=' * 70)

    shuttlenet_script = os.path.join(project_path, 'Agent', 'shuttlenet.py')
    cmd = [
        'python', shuttlenet_script,
        '--mode', 'train',
        '--experiment', experiment_type,
        '--merged_csv', config['paths']['merged_csv'],
        '--player_map', config['paths']['player_map'],
        '--split', split_path,
        '--data_dir', data_dir,
        '--output', output_path,
        '--device', device,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--encode_length', str(encode_length),
        '--max_ball_round', str(max_ball_round),
        '--shot_dim', str(shot_dim),
        '--area_dim', str(area_dim),
        '--player_dim', str(player_dim),
        '--encode_dim', str(encode_dim),
        '--area_num', str(area_num),
        '--dropout', str(dropout),
    ]
    if reprepare:
        cmd.append('--reprepare')

    print(f"\n執行命令:\n{' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=project_path)
    if result.returncode == 0:
        print(f"\n✅ ShuttleNet 訓練完成! 模型已儲存至: {output_path}")
        return True

    print('\n❌ ShuttleNet 訓練失敗')
    return False


def list_available_data():
    config = load_config()
    print('\n可用的 ShuttleNet 訓練資料:')
    print('-' * 70)
    for exp_type, data_dir in config['paths']['shuttlenet']['data'].items():
        sample_path = os.path.join(data_dir, 'train_samples.pkl')
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                payload = pickle.load(f)
            print(f"  ✓ {exp_type:10s}: {len(payload.get('samples', []))} samples - {data_dir}")
        else:
            print(f'  ✗ {exp_type:10s}: 資料不存在 - {data_dir}')
    print('-' * 70)


def list_trained_models():
    config = load_config()
    weight_dir = config['paths']['shuttlenet']['weight_dir']
    print('\n已訓練的 ShuttleNet 模型:')
    print('-' * 70)
    if os.path.exists(weight_dir):
        models = [filename for filename in os.listdir(weight_dir) if filename.endswith('.pth')]
        if models:
            for model in sorted(models):
                model_path = os.path.join(weight_dir, model)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f'  - {model} ({size_mb:.2f} MB)')
        else:
            print('  (無模型)')
    else:
        print(f'  (目錄不存在: {weight_dir})')
    print('-' * 70)


def main():
    parser = argparse.ArgumentParser(
        description='訓練 ShuttleNet 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python Scripts/train_shuttlenet.py --experiment norm
  python Scripts/train_shuttlenet.py --experiment shifted
  python Scripts/train_shuttlenet.py --experiment norm --epochs 20 --batch-size 128
  python Scripts/train_shuttlenet.py --list-data
  python Scripts/train_shuttlenet.py --list-models
        """,
    )
    parser.add_argument('--experiment', choices=['norm', 'shifted'], help='實驗類型')
    parser.add_argument('--epochs', type=int, help='訓練 epoch 數')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='學習率')
    parser.add_argument('--output-name', help='自訂輸出檔名（在 shuttlenet weight_dir 下）')
    parser.add_argument('--data-dir', help='自訂 ShuttleNet 訓練資料目錄')
    parser.add_argument('--encode-length', type=int, help='history encode 長度')
    parser.add_argument('--max-ball-round', type=int, help='最大 rally 長度 positional table 設定')
    parser.add_argument('--shot-dim', type=int, help='shot embedding dim')
    parser.add_argument('--area-dim', type=int, help='area embedding dim')
    parser.add_argument('--player-dim', type=int, help='player embedding dim')
    parser.add_argument('--encode-dim', type=int, help='hidden encode dim')
    parser.add_argument('--area-num', type=int, help='2D Gaussian param dim')
    parser.add_argument('--dropout', type=float, help='dropout')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='運算裝置')
    parser.add_argument('--reprepare', action='store_true', help='訓練前重建 train_samples.pkl')
    parser.add_argument('--list-data', action='store_true', help='列出可用訓練資料')
    parser.add_argument('--list-models', action='store_true', help='列出已訓練模型')

    args = parser.parse_args()

    if args.list_data:
        list_available_data()
        return 0
    if args.list_models:
        list_trained_models()
        return 0
    if not args.experiment:
        parser.print_help()
        print('\n❌ 錯誤: 必須指定 --experiment，或使用 --list-data / --list-models')
        return 1

    success = train_shuttlenet(
        experiment_type=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_name=args.output_name,
        data_dir=args.data_dir,
        encode_length=args.encode_length,
        max_ball_round=args.max_ball_round,
        shot_dim=args.shot_dim,
        area_dim=args.area_dim,
        player_dim=args.player_dim,
        encode_dim=args.encode_dim,
        area_num=args.area_num,
        dropout=args.dropout,
        device=args.device,
        reprepare=args.reprepare,
    )
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
