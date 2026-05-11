#!/usr/bin/env python3
"""
BC 模型訓練腳本
使用 config.yaml 中的設定進行訓練

使用方式:
    python Scripts/train_bc.py --experiment norm       # 在 norm 資料上訓練
    python Scripts/train_bc.py --experiment shifted    # 在 shifted 資料上訓練
    python Scripts/train_bc.py --experiment norm --epochs 200 --lr 0.0005
"""

import argparse
import os
import sys
import subprocess
import yaml

# 加入專案路徑
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


def load_config():
    """載入設定檔"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_bc(experiment_type, epochs=None, batch_size=None, lr=None, output_name=None,
             distribution='deterministic', gmm_num=None):
    """
    訓練 BC 模型
    
    Args:
        experiment_type: 'norm' 或 'shifted'
        epochs: 訓練輪數（可選，預設使用 config）
        batch_size: 批次大小（可選，預設使用 config）
        lr: 學習率（可選，預設使用 config）
        output_name: 自訂輸出檔名（可選）
    """
    config = load_config()
    
    # 取得路徑
    if experiment_type == 'norm':
        data_dir = config['paths']['bc']['data']['norm']
        default_output = config['paths']['bc']['models']['norm']
    elif experiment_type == 'shifted':
        data_dir = config['paths']['bc']['data']['shifted']
        default_output = config['paths']['bc']['models']['shifted']
    else:
        print(f"❌ 不支援的實驗類型: {experiment_type}")
        print("   支援的類型: norm, shifted")
        return False
    
    # 決定輸出路徑
    if output_name:
        output_path = os.path.join(config['paths']['bc']['weight_dir'], output_name)
    else:
        output_path = default_output
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 設定輸出分布
    distribution = distribution or config['bc_training'].get('distribution', 'deterministic')
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
    
    # 檢查資料是否存在
    state_path = os.path.join(data_dir, 'train_state.pkl')
    action_path = os.path.join(data_dir, 'train_action.pkl')
    
    if not os.path.exists(state_path) or not os.path.exists(action_path):
        print(f"❌ 訓練資料不存在: {data_dir}")
        print(f"   請先執行: python Scripts/prepare_all_data.py --{experiment_type} --bc")
        return False
    
    print("="*70)
    print(f"BC 模型訓練 - {experiment_type.upper()}")
    print("="*70)
    print(f"資料來源: {data_dir}")
    print(f"輸出模型: {output_path}")
    print(f"訓練參數:")
    print(f"  - Epochs:     {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {lr}")
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
        '--gmm_num', str(gmm_num)
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
    
    for exp_type in ['norm', 'shifted']:
        data_dir = config['paths']['bc']['data'][exp_type]
        state_path = os.path.join(data_dir, 'train_state.pkl')
        action_path = os.path.join(data_dir, 'train_action.pkl')
        
        if os.path.exists(state_path) and os.path.exists(action_path):
            import pickle
            with open(state_path, 'rb') as f:
                state_data = pickle.load(f)
            print(f"  ✓ {exp_type:10s}: {len(state_data)} 樣本 - {data_dir}")
        else:
            print(f"  ✗ {exp_type:10s}: 資料不存在 - {data_dir}")
    
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
  # 在 norm 資料上訓練
  python Scripts/train_bc.py --experiment norm
  
  # 在 shifted 資料上訓練
  python Scripts/train_bc.py --experiment shifted
  
  # 自訂訓練參數
  python Scripts/train_bc.py --experiment shifted --epochs 200 --lr 0.0005
  
  # 自訂輸出檔名
  python Scripts/train_bc.py --experiment shifted --output-name my_model.pth
  
  # 查看可用的資料和模型
  python Scripts/train_bc.py --list-data
  python Scripts/train_bc.py --list-models
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['norm', 'shifted'],
        help='實驗類型: norm (train_test_split) 或 shifted (style_shift)'
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
        help='自訂輸出檔名（預設: norm_model.pth 或 shifted_model.pth）'
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
    
    # 檢查必要參數
    if not args.experiment:
        parser.print_help()
        print("\n❌ 錯誤: 必須指定 --experiment (norm 或 shifted)")
        return 1
    
    # 執行訓練
    success = train_bc(
        experiment_type=args.experiment,
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
