#!/usr/bin/env python3
"""
完整實驗流程腳本
自動執行從資料準備到評估的所有步驟

使用方式:
    python Scripts/run_full_pipeline.py --experiment norm       # 完整 norm 實驗
    python Scripts/run_full_pipeline.py --experiment shifted    # 完整 shifted 實驗
    python Scripts/run_full_pipeline.py --all                   # 兩種實驗都執行
"""

import argparse
import os
import sys
import subprocess
import yaml
import time

# 加入專案路徑
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


def load_config():
    """載入設定檔"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_script(script_name, *args):
    """
    執行指定的腳本
    
    Args:
        script_name: 腳本名稱（不含路徑）
        *args: 傳遞給腳本的參數
    
    Returns:
        (success: bool, duration: float)
    """
    script_path = os.path.join(project_path, 'Scripts', script_name)
    cmd = ['python', script_path] + list(args)
    
    print(f"\n{'='*70}")
    print(f"執行: {script_name} {' '.join(args)}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=project_path)
    duration = time.time() - start_time
    
    success = result.returncode == 0
    
    if success:
        print(f"\n✅ {script_name} 完成 (耗時: {duration:.1f}s)")
    else:
        print(f"\n❌ {script_name} 失敗 (耗時: {duration:.1f}s)")
    
    return success, duration


def run_full_pipeline(experiment_type, prepare_data=True, train=True, evaluate=True, epochs=None):
    """
    執行完整實驗流程
    
    Args:
        experiment_type: 'norm' 或 'shifted'
        prepare_data: 是否準備資料
        train: 是否訓練模型
        evaluate: 是否評估模型
        epochs: 訓練輪數（可選）
    """
    config = load_config()
    
    print("\n" + "="*70)
    print(f"完整實驗流程 - {experiment_type.upper()}")
    print("="*70)
    print(f"準備資料: {'是' if prepare_data else '否'}")
    print(f"訓練模型: {'是' if train else '否'}")
    print(f"評估模型: {'是' if evaluate else '否'}")
    if epochs:
        print(f"訓練輪數: {epochs}")
    print("="*70 + "\n")
    
    total_start_time = time.time()
    step_results = []
    
    # Step 1: 準備資料
    if prepare_data:
        # 如果是 shifted，先創建 split
        if experiment_type == 'shifted':
            success, duration = run_script('create_style_shift_split.py')
            step_results.append(('Create Split', success, duration))
            if not success:
                print("\n❌ 創建 split 失敗，中止流程")
                return False
        
        # 準備 BC 和 NaiveRAG 資料
        success, duration = run_script(
            'prepare_all_data.py',
            f'--{experiment_type}'
        )
        step_results.append(('Prepare Data', success, duration))
        if not success:
            print("\n❌ 資料準備失敗，中止流程")
            return False
    
    # Step 2: 訓練模型
    if train:
        train_args = ['--experiment', experiment_type]
        if epochs:
            train_args.extend(['--epochs', str(epochs)])
        
        success, duration = run_script('train_bc.py', *train_args)
        step_results.append(('Train BC', success, duration))
        if not success:
            print("\n❌ 訓練失敗，中止流程")
            return False
    
    # Step 3: 評估 BC 模型
    if evaluate:
        success, duration = run_script(
            'eval_bc.py',
            '--experiment', experiment_type
        )
        step_results.append(('Evaluate BC', success, duration))
        
        # Step 4: 評估 TimeRAG 模型
        success, duration = run_script(
            'eval_timerag.py',
            '--test_split', config['paths'][f'{experiment_type}_split'],
            '--merged_csv', config['paths']['merged_csv'],
            '--player_map', config['paths']['player_map'],
            '--database_dir', config['paths']['timerag']['data'][experiment_type],
            '--output', config['paths']['timerag']['results'][experiment_type]
        )
        step_results.append(('Evaluate TimeRAG', success, duration))
    
    # 輸出總結
    total_duration = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("實驗流程完成!")
    print("="*70)
    print("\n各步驟結果:")
    print("-"*70)
    
    for step_name, success, duration in step_results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {step_name:20s} {status:10s} ({duration:6.1f}s)")
    
    print("-"*70)
    print(f"  {'總計':20s} {'':10s} ({total_duration:6.1f}s)")
    print("="*70)
    
    # 顯示輸出位置
    print("\n輸出檔案:")
    print("-"*70)
    
    if train:
        model_path = config['paths']['bc']['models'][experiment_type]
        print(f"  BC 模型: {model_path}")
    
    if evaluate:
        result_path = config['paths']['bc']['results'][experiment_type]
        print(f"  BC 評估結果: {result_path}")
        timerag_result_path = config['paths']['timerag']['results'][experiment_type]
        print(f"  TimeRAG 評估結果: {timerag_result_path}")
    
    print("-"*70)
    
    return all(success for _, success, _ in step_results)


def run_both_experiments(epochs=None):
    """執行兩種實驗（norm 和 shifted）"""
    print("\n" + "="*70)
    print("執行兩種實驗 (Norm + Shifted)")
    print("="*70 + "\n")
    
    results = {}
    
    for exp_type in ['norm', 'shifted']:
        success = run_full_pipeline(
            experiment_type=exp_type,
            prepare_data=True,
            train=True,
            evaluate=True,
            epochs=epochs
        )
        results[exp_type] = success
        
        if not success:
            print(f"\n⚠️ {exp_type} 實驗失敗，繼續下一個實驗...\n")
    
    # 總結
    print("\n" + "="*70)
    print("兩種實驗結果總結")
    print("="*70)
    
    for exp_type, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {exp_type.upper():15s} {status}")
    
    print("="*70)
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description='執行完整實驗流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 執行單一實驗（準備資料、訓練、評估）
  python Scripts/run_full_pipeline.py --experiment norm
  python Scripts/run_full_pipeline.py --experiment shifted
  
  # 執行兩種實驗
  python Scripts/run_full_pipeline.py --all
  
  # 只訓練和評估（不重新準備資料）
  python Scripts/run_full_pipeline.py --experiment shifted --no-prepare
  
  # 只評估（使用已訓練的模型）
  python Scripts/run_full_pipeline.py --experiment shifted --no-prepare --no-train
  
  # 自訂訓練參數
  python Scripts/run_full_pipeline.py --experiment shifted --epochs 200
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['norm', 'shifted'],
        help='實驗類型: norm 或 shifted'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='執行兩種實驗（norm 和 shifted）'
    )
    parser.add_argument(
        '--no-prepare',
        action='store_true',
        help='跳過資料準備步驟'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='跳過訓練步驟'
    )
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='跳過評估步驟'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='訓練輪數（覆蓋 config 設定）'
    )
    
    args = parser.parse_args()
    
    # 檢查參數
    if not args.experiment and not args.all:
        parser.print_help()
        print("\n❌ 錯誤: 必須指定 --experiment 或 --all")
        return 1
    
    # 決定執行哪些步驟
    prepare_data = not args.no_prepare
    train = not args.no_train
    evaluate = not args.no_eval
    
    if args.all:
        success = run_both_experiments(epochs=args.epochs)
    else:
        success = run_full_pipeline(
            experiment_type=args.experiment,
            prepare_data=prepare_data,
            train=train,
            evaluate=evaluate,
            epochs=args.epochs
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
