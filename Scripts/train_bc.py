#!/usr/bin/env python3
"""
BC 模型訓練腳本
使用 config.yaml 中的設定進行訓練

使用方式:
    python Scripts/train_bc.py
    python Scripts/train_bc.py --epochs 200 --lr 0.0005
"""

import argparse
import os
import sys
import subprocess
import yaml

# 加入專案路徑
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Utils.runtime_report import algorithm_runtime_rows, print_runtime_box


def load_config():
    """載入設定檔"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_bc(epochs=None, batch_size=None, lr=None, output_name=None,
             distribution='deterministic', gmm_num=None):
    """
    訓練 BC 模型
    
    Args:
        epochs: 訓練輪數（可選，預設使用 config）
        batch_size: 批次大小（可選，預設使用 config）
        lr: 學習率（可選，預設使用 config）
        output_name: 自訂輸出檔名（可選）
    """
    config = load_config()
    
    data_dir = config['paths']['bc']['data']
    default_output = config['paths']['bc']['model']
    
    distribution = distribution or config['bc_training'].get('distribution', 'deterministic')

    # Keep correlated Gaussian checkpoints separate from the default GMM models.
    if output_name:
        output_path = os.path.join(config['paths']['bc']['weight_dir'], output_name)
    elif distribution == 'gaussian':
        output_path = default_output
    else:
        output_path = default_output
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 設定輸出分布
    if distribution not in ('deterministic', 'gaussian', 'gmm'):
        print(f"❌ 不支援的 distribution: {distribution}")
        print("   支援的類型: deterministic, gaussian, gmm")
        return False

    # 取得訓練參數
    bc_config = config['bc_training']
    epochs = epochs or bc_config['epochs']
    batch_size = batch_size or bc_config['batch_size']
    lr = lr or bc_config['learning_rate']
    gmm_num = gmm_num or bc_config.get('gmm_num', 5)
    input_dim = bc_config['input_dim']
    hidden_dim = bc_config['hidden_dim']
    player_id_len = bc_config['player_id_len']
    shot_type_len = bc_config['shot_type_len']

    print_runtime_box(
        "Runtime Parameters: train_bc",
        algorithm_runtime_rows(
            config,
            "bc",
            experiment='rally',
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            distribution=distribution,
            gmm_num=gmm_num,
        ) + [
            ("bc.input_dim", input_dim),
            ("bc.hidden_dim", hidden_dim),
            ("bc.player_id_len", player_id_len),
            ("bc.shot_type_len", shot_type_len),
            ("data_dir", data_dir),
            ("output_path", output_path),
        ],
    )
    
    # 檢查資料是否存在
    state_path = os.path.join(data_dir, 'train_state.pkl')
    action_path = os.path.join(data_dir, 'train_action.pkl')
    
    if not os.path.exists(state_path) or not os.path.exists(action_path):
        print(f"❌ 訓練資料不存在: {data_dir}")
        print("   請先執行: python Scripts/prepare_all_data.py --bc")
        return False
    
    print("="*70)
    print("BC 模型訓練 - RALLY")
    print("="*70)
    print(f"資料來源: {data_dir}")
    print(f"輸出模型: {output_path}")
    print(f"訓練參數:")
    print(f"  - Epochs:     {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Input Dimension: {input_dim}")
    print(f"  - Hidden Dimension: {hidden_dim}")
    print(f"  - Player ID Count: {player_id_len}")
    print(f"  - Shot Type Count: {shot_type_len}")
    print(f"  - Distribution: {distribution}")
    if distribution == 'gmm':
        print(f"  - GMM Components: {gmm_num}")
    print("="*70)
    
    # 執行訓練
    bc_script = os.path.join(project_path, 'Agent', 'bc.py')
    
    cmd = [
        'python', bc_script,
        '--mode', 'train',
        '--data_dir', data_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--output', output_path,
        '--distribution', distribution,
        '--gmm_num', str(gmm_num),
        '--input_dim', str(input_dim),
        '--hidden_dim', str(hidden_dim),
        '--player_id_len', str(player_id_len),
        '--shot_type_len', str(shot_type_len)
    ]
    
    print(f"\n執行命令:\n{' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=project_path)
    
    if result.returncode == 0:
        print(f"\n✅ 訓練完成! 模型已儲存至: {output_path}")
        return True
    else:
        print(f"\n❌ 訓練失敗")
        return False


def list_available_data():
    """列出可用的訓練資料"""
    config = load_config()
    
    print("\n可用的訓練資料:")
    print("-"*70)
    
    data_dir = config['paths']['bc']['data']
    state_path = os.path.join(data_dir, 'train_state.pkl')
    action_path = os.path.join(data_dir, 'train_action.pkl')
    if os.path.exists(state_path) and os.path.exists(action_path):
        import pickle
        with open(state_path, 'rb') as f:
            state_data = pickle.load(f)
        print(f"  ✓ rally: {len(state_data)} 樣本 - {data_dir}")
    else:
        print(f"  ✗ rally: 資料不存在 - {data_dir}")
    
    print("-"*70)


def list_trained_models():
    """列出已訓練的模型"""
    config = load_config()
    weight_dir = config['paths']['bc']['weight_dir']
    
    print("\n已訓練的模型:")
    print("-"*70)
    
    if os.path.exists(weight_dir):
        models = [f for f in os.listdir(weight_dir) if f.endswith('.pth')]
        if models:
            for model in sorted(models):
                model_path = os.path.join(weight_dir, model)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"  - {model} ({size_mb:.2f} MB)")
        else:
            print("  (無模型)")
    else:
        print(f"  (目錄不存在: {weight_dir})")
    
    print("-"*70)


def main():
    parser = argparse.ArgumentParser(
        description='訓練 BC 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 在 rally-level 資料上訓練
  python Scripts/train_bc.py
  
  # 自訂訓練參數
  python Scripts/train_bc.py --epochs 200 --lr 0.0005
  
  # 自訂輸出檔名
  python Scripts/train_bc.py --output-name my_model.pth
  
  # 查看可用的資料和模型
  python Scripts/train_bc.py --list-data
  python Scripts/train_bc.py --list-models
        """
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='訓練輪數（覆蓋 config 設定）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='批次大小（覆蓋 config 設定）'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='學習率（覆蓋 config 設定）'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        help='自訂輸出檔名（Gaussian 預設使用獨立的 correlated_gaussian_model 檔名）'
    )
    parser.add_argument(
        '--distribution',
        type=str,
        choices=['deterministic', 'gaussian', 'gmm'],
        help='動作輸出模式: deterministic、gaussian 或 gmm（預設讀取 config）'
    )
    parser.add_argument(
        '--gmm-num',
        type=int,
        help='GMM component 數量（僅 distribution=gmm，預設讀取 config）'
    )
    parser.add_argument(
        '--list-data',
        action='store_true',
        help='列出可用的訓練資料'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出已訓練的模型'
    )
    
    args = parser.parse_args()
    config = load_config()
    
    # 如果只列出資訊
    if args.list_data:
        list_available_data()
        return 0
    
    if args.list_models:
        list_trained_models()
        return 0
    
    # 執行訓練
    success = train_bc(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_name=args.output_name,
        distribution=args.distribution or config['bc_training'].get('distribution', 'deterministic'),
        gmm_num=args.gmm_num or config['bc_training'].get('gmm_num', 5)
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
