#!/usr/bin/env python3
"""
RallyNetV2 training script.

Usage:
    python Scripts/train_rallynetv2.py --experiment norm
    python Scripts/train_rallynetv2.py --experiment norm --num-iters 100 --pause-every 10
    python Scripts/train_rallynetv2.py --experiment norm --output-dir Experiments/rallynetv2/weight_no_logvar
"""

import argparse
import os
import pickle
import sys

import yaml


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


REQUIRED_DATA_FILES = [
    'train_player_traj.pkl',
    'train_opponent_traj.pkl',
    'train_player_mask.pkl',
    'train_opponent_mask.pkl',
]


def load_config():
    """Load centralized experiment config."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_experiment_data_dir(config, experiment_type, data_dir=None):
    if data_dir:
        return data_dir
    try:
        return config['paths']['rallynetv2']['data'][experiment_type]
    except KeyError as exc:
        raise ValueError(f'Unsupported experiment type for RallyNetV2: {experiment_type}') from exc


def resolve_experiment_output_dir(config, experiment_type, output_dir=None):
    if output_dir:
        return output_dir
    try:
        return config['paths']['rallynetv2']['weight'][experiment_type]
    except KeyError as exc:
        raise ValueError(f'Unsupported experiment type for RallyNetV2: {experiment_type}') from exc


def resolve_experiment_hyperparameters(config, experiment_type, hyperparameters_output=None):
    if hyperparameters_output:
        return hyperparameters_output
    try:
        return config['paths']['rallynetv2']['hyperparameters'][experiment_type]
    except KeyError as exc:
        raise ValueError(f'Unsupported experiment type for RallyNetV2: {experiment_type}') from exc


def has_required_data(data_dir):
    return all(os.path.exists(os.path.join(data_dir, filename)) for filename in REQUIRED_DATA_FILES)


def train_rallynetv2(experiment_type, num_iters=None, batch_size=None, lr=None,
                     output_dir=None, hyperparameters_output=None, data_dir=None,
                     latent_size=None, context_size=None, hidden_size=None,
                     lr_gamma=None, kl_anneal_iters=None, pause_every=None,
                     noise_std=None, adjoint=None, method=None, split_ratio=None,
                     gmm_num=None, min_sigma=None, load=False, load_step=0):
    """Train RallyNetV2 using Agent.rallynetv2.main."""
    config = load_config()
    rallynetv2_paths = config['paths']['rallynetv2']
    train_config = config['rallynetv2_training']

    resolved_data_dir = resolve_experiment_data_dir(config, experiment_type, data_dir)
    output_dir = resolve_experiment_output_dir(config, experiment_type, output_dir)
    hyperparameters_output = resolve_experiment_hyperparameters(config, experiment_type, hyperparameters_output)

    num_iters = num_iters if num_iters is not None else train_config['num_iters']
    batch_size = batch_size if batch_size is not None else train_config['batch_size']
    lr = lr if lr is not None else train_config['learning_rate']
    latent_size = latent_size if latent_size is not None else train_config['latent_size']
    context_size = context_size if context_size is not None else train_config['context_size']
    hidden_size = hidden_size if hidden_size is not None else train_config['hidden_size']
    lr_gamma = lr_gamma if lr_gamma is not None else train_config['lr_gamma']
    kl_anneal_iters = kl_anneal_iters if kl_anneal_iters is not None else train_config['kl_anneal_iters']
    pause_every = pause_every if pause_every is not None else train_config['pause_every']
    noise_std = noise_std if noise_std is not None else train_config['noise_std']
    adjoint = adjoint if adjoint is not None else train_config['adjoint']
    method = method or train_config['method']
    split_ratio = split_ratio if split_ratio is not None else train_config['split_ratio']
    gmm_num = gmm_num if gmm_num is not None else train_config['gmm_num']
    min_sigma = min_sigma if min_sigma is not None else train_config['min_sigma']

    if not has_required_data(resolved_data_dir):
        print(f"❌ RallyNetV2 訓練資料不存在或不完整: {resolved_data_dir}")
        print("   請先執行:")
        print("   python Preprocessing/baseline_prepare.py --rallynetv2")
        return False

    if load:
        checkpoint_path = os.path.join(output_dir, f'gen_e_{load_step}.trc')
        if not os.path.exists(checkpoint_path):
            print(f"❌ 指定續訓 checkpoint 不存在: {checkpoint_path}")
            return False

    os.makedirs(output_dir, exist_ok=True)
    hyperparameters_dir = os.path.dirname(hyperparameters_output)
    if hyperparameters_dir:
        os.makedirs(hyperparameters_dir, exist_ok=True)

    print('=' * 70)
    print(f'RallyNetV2 模型訓練 - {experiment_type.upper()}')
    print('=' * 70)
    print(f'資料來源: {resolved_data_dir}')
    print(f'輸出目錄: {output_dir}')
    print(f'Hyperparameters: {hyperparameters_output}')
    print('訓練參數:')
    print(f'  - Num Iters:       {num_iters}')
    print(f'  - Batch Size:      {batch_size}')
    print(f'  - Learning Rate:   {lr}')
    print(f'  - Latent Size:     {latent_size}')
    print(f'  - Context Size:    {context_size}')
    print(f'  - Hidden Size:     {hidden_size}')
    print(f'  - KL Anneal Iters: {kl_anneal_iters}')
    print(f'  - Pause Every:     {pause_every}')
    print(f'  - Method:          {method}')
    print(f'  - GMM Components:  {gmm_num}')
    print(f'  - Min Sigma:       {min_sigma}')
    if load:
        print(f'  - Resume Step:     {load_step}')
    print('=' * 70)

    os.chdir(project_path)
    from Agent.rallynetv2 import main as rallynetv2_main

    rallynetv2_main(
        batch_size=batch_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        lr_init=lr,
        lr_gamma=lr_gamma,
        num_iters=num_iters,
        kl_anneal_iters=kl_anneal_iters,
        pause_every=pause_every,
        noise_std=noise_std,
        adjoint=adjoint,
        train_dir=output_dir,
        method=method,
        load=load,
        load_step=load_step,
        split_ratio=split_ratio,
        gmm_num=gmm_num,
        min_sigma=min_sigma,
        data_dir=resolved_data_dir,
        hyperparameters_output=hyperparameters_output,
    )

    print(f"\n✅ 訓練完成! 權重已儲存至: {output_dir}")
    print(f"✅ Hyperparameters 已儲存至: {hyperparameters_output}")
    return True


def list_available_data():
    """List RallyNetV2 training data directories."""
    config = load_config()

    print('\n可用的 RallyNetV2 訓練資料:')
    print('-' * 70)

    for exp_type, data_dir in config['paths']['rallynetv2']['data'].items():
        if has_required_data(data_dir):
            player_traj_path = os.path.join(data_dir, 'train_player_traj.pkl')
            with open(player_traj_path, 'rb') as f:
                player_data = pickle.load(f)
            print(f'  ✓ {exp_type:10s}: {len(player_data)} rallies - {data_dir}')
        else:
            print(f'  ✗ {exp_type:10s}: 資料不存在或不完整 - {data_dir}')

    print('-' * 70)


def list_trained_models():
    """List saved RallyNetV2 checkpoints."""
    config = load_config()

    print('\n已訓練的 RallyNetV2 模型:')
    print('-' * 70)

    for exp_type, weight_dir in config['paths']['rallynetv2']['weight'].items():
        print(f'  {exp_type}: {weight_dir}')
        if os.path.exists(weight_dir):
            models = [f for f in os.listdir(weight_dir) if f.endswith('.trc')]
            if models:
                for model in sorted(models):
                    model_path = os.path.join(weight_dir, model)
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f'    - {model} ({size_mb:.2f} MB)')
            else:
                print('    (無模型)')
        else:
            print('    (目錄不存在)')

    print('-' * 70)


def main():
    parser = argparse.ArgumentParser(
        description='訓練 RallyNetV2 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 在 norm 資料上訓練
  python Scripts/train_rallynetv2.py --experiment norm

  # 快速測試訓練流程
  python Scripts/train_rallynetv2.py --experiment norm --num-iters 100 --pause-every 10

  # 自訂輸出目錄
  python Scripts/train_rallynetv2.py --experiment norm --output-dir Experiments/rallynetv2/weight/custom

  # 查看可用資料和模型
  python Scripts/train_rallynetv2.py --list-data
  python Scripts/train_rallynetv2.py --list-models
        """
    )

    parser.add_argument('--experiment', type=str, choices=['norm', 'shifted'],
                        help='實驗類型: norm 或 shifted')
    parser.add_argument('--num-iters', type=int,
                        help='訓練 iteration 數（覆蓋 config 設定）')
    parser.add_argument('--batch-size', type=int,
                        help='批次大小（覆蓋 config 設定）')
    parser.add_argument('--lr', type=float,
                        help='學習率（覆蓋 config 設定）')
    parser.add_argument('--output-dir', type=str,
                        help='權重輸出目錄（預設讀取 config）')
    parser.add_argument('--hyperparameters-output', type=str,
                        help='hyperparameters.pkl 輸出路徑（預設讀取 config）')
    parser.add_argument('--data-dir', type=str,
                        help='自訂 RallyNetV2 訓練資料目錄')
    parser.add_argument('--latent-size', type=int,
                        help='latent size（覆蓋 config 設定）')
    parser.add_argument('--context-size', type=int,
                        help='context size（覆蓋 config 設定）')
    parser.add_argument('--hidden-size', type=int,
                        help='hidden size（覆蓋 config 設定）')
    parser.add_argument('--lr-gamma', type=float,
                        help='learning-rate scheduler gamma（覆蓋 config 設定）')
    parser.add_argument('--kl-anneal-iters', type=int,
                        help='KL annealing iterations（覆蓋 config 設定）')
    parser.add_argument('--pause-every', type=int,
                        help='每幾步儲存 checkpoint（覆蓋 config 設定）')
    parser.add_argument('--noise-std', type=float,
                        help='SDE reconstruction likelihood noise std（覆蓋 config 設定）')
    parser.add_argument('--adjoint', action='store_true', default=None,
                        help='使用 torchsde adjoint solver')
    parser.add_argument('--method', type=str,
                        help='SDE solver method（預設讀取 config）')
    parser.add_argument('--split-ratio', type=float,
                        help='保留原 main 參數，目前 dataset 建立未實際切分')
    parser.add_argument('--gmm-num', type=int,
                        help='GMM component 數量（覆蓋 config 設定）')
    parser.add_argument('--min-sigma', type=float,
                        help='GMM minimum sigma（覆蓋 config 設定）')
    parser.add_argument('--load', action='store_true',
                        help='從 output-dir/gen_e_<load-step>.trc 續訓')
    parser.add_argument('--load-step', type=int, default=0,
                        help='續訓 checkpoint step')
    parser.add_argument('--list-data', action='store_true',
                        help='列出可用的 RallyNetV2 訓練資料')
    parser.add_argument('--list-models', action='store_true',
                        help='列出已訓練的 RallyNetV2 模型')

    args = parser.parse_args()

    if args.list_data:
        list_available_data()
        return 0

    if args.list_models:
        list_trained_models()
        return 0

    if not args.experiment:
        parser.print_help()
        print('\n❌ 請指定 --experiment，或使用 --list-data / --list-models')
        return 1

    success = train_rallynetv2(
        experiment_type=args.experiment,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        hyperparameters_output=args.hyperparameters_output,
        data_dir=args.data_dir,
        latent_size=args.latent_size,
        context_size=args.context_size,
        hidden_size=args.hidden_size,
        lr_gamma=args.lr_gamma,
        kl_anneal_iters=args.kl_anneal_iters,
        pause_every=args.pause_every,
        noise_std=args.noise_std,
        adjoint=args.adjoint,
        method=args.method,
        split_ratio=args.split_ratio,
        gmm_num=args.gmm_num,
        min_sigma=args.min_sigma,
        load=args.load,
        load_step=args.load_step,
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
