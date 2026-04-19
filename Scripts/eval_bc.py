#!/usr/bin/env python3
"""
BC 模型評估腳本
評估模型在 norm 或 shifted test 資料上的表現

使用方式:
    python Scripts/eval_bc.py --experiment norm       # 在 norm test 上評估 norm 模型
    python Scripts/eval_bc.py --experiment shifted    # 在 shifted test 上評估 shifted 模型
    python Scripts/eval_bc.py --train-on norm --test-on shifted  # 跨實驗評估
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


def eval_bc(train_experiment, test_experiment, model_name=None, device=None, skip_first_n=None):
    """
    評估 BC 模型
    
    Args:
        train_experiment: 模型訓練時使用的實驗類型 ('norm' 或 'shifted')
        test_experiment: 要測試的實驗類型 ('norm' 或 'shifted')
        model_name: 自訂模型檔名（可選）
        device: 運算裝置（可選）
        skip_first_n: 跳過前 n 個 shots（可選）
    """
    config = load_config()
    
    # 取得路徑
    weight_dir = config['paths']['bc']['weight_dir']
    results_dir = config['paths']['bc']['results_dir']
    
    # 決定模型路徑
    if model_name:
        model_path = os.path.join(weight_dir, model_name)
    else:
        model_path = config['paths']['bc']['models'][train_experiment]
    
    # 決定 test split
    test_split = config['paths'][f'{test_experiment}_split']
    
    # 決定輸出路徑
    if train_experiment == test_experiment:
        output_name = f"{test_experiment}_eval.json"
    else:
        output_name = f"{train_experiment}_on_{test_experiment}_eval.json"
    
    output_path = os.path.join(results_dir, output_name)
    
    # 取得評估參數
    device = device or config['evaluation']['device']
    skip_first_n = skip_first_n if skip_first_n is not None else config['evaluation']['skip_first_n']
    
    # 檢查必要檔案
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print(f"   請先執行: python Scripts/train_bc.py --experiment {train_experiment}")
        return False
    
    if not os.path.exists(test_split):
        if test_experiment == 'shifted':
            print(f"❌ Test split 不存在: {test_split}")
            print(f"   請先執行: python Scripts/create_style_shift_split.py")
        else:
            print(f"❌ Test split 不存在: {test_split}")
        return False
    
    # 確保輸出目錄存在
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    if train_experiment == test_experiment:
        print(f"BC 模型評估 - {test_experiment.upper()}")
    else:
        print(f"BC 模型評估 - {train_experiment.upper()} 模型在 {test_experiment.upper()} 資料上")
    print("="*70)
    print(f"模型: {model_path}")
    print(f"測試資料: {test_split}")
    print(f"輸出結果: {output_path}")
    print(f"評估參數:")
    print(f"  - Device: {device}")
    print(f"  - Skip First N: {skip_first_n}")
    print("="*70)
    
    # 執行評估
    bc_script = os.path.join(project_path, 'Agent', 'bc.py')
    
    cmd = [
        'python', bc_script,
        '--mode', 'eval',
        '--model_path', model_path,
        '--test_split', test_split,
        '--merged_csv', config['paths']['merged_csv'],
        '--player_map', config['paths']['player_map'],
        '--device', device,
        '--eval_output', output_path
    ]
    
    if skip_first_n > 0:
        cmd.extend(['--skip_first_n', str(skip_first_n)])
    
    print(f"\n執行命令:\n{' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=project_path)
    
    if result.returncode == 0:
        print(f"\n✅ 評估完成! 結果已儲存至: {output_path}")
        return True
    else:
        print(f"\n❌ 評估失敗")
        return False


def list_available_models():
    """列出可用的模型和評估結果"""
    config = load_config()
    weight_dir = config['paths']['bc']['weight_dir']
    results_dir = config['paths']['bc']['results_dir']
    
    print("\n可用的模型:")
    print("-"*70)
    
    if os.path.exists(weight_dir):
        models = [f for f in os.listdir(weight_dir) if f.endswith('.pth')]
        if models:
            for model in sorted(models):
                model_path = os.path.join(weight_dir, model)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"  ✓ {model} ({size_mb:.2f} MB)")
        else:
            print("  (無模型)")
    else:
        print(f"  (目錄不存在)")
    
    print("\n評估結果:")
    print("-"*70)
    
    if os.path.exists(results_dir):
        results = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if results:
            for result_file in sorted(results):
                print(f"  ✓ {result_file}")
        else:
            print("  (無結果)")
    else:
        print(f"  (目錄不存在)")
    
    print("-"*70)


def view_result(result_name=None):
    """查看評估結果"""
    config = load_config()
    results_dir = config['paths']['bc']['results_dir']
    
    if result_name:
        result_path = os.path.join(results_dir, result_name)
    else:
        # 找最新的結果
        if os.path.exists(results_dir):
            json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if json_files:
                result_path = os.path.join(results_dir, sorted(json_files)[-1])
            else:
                print("❌ 沒有找到評估結果")
                return False
        else:
            print("❌ 結果目錄不存在")
            return False
    
    if not os.path.exists(result_path):
        print(f"❌ 結果檔案不存在: {result_path}")
        return False
    
    import json
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n評估結果: {os.path.basename(result_path)}")
    print("="*70)
    
    # 計算平均
    total_land_dtw = []
    total_move_dtw = []
    total_edit_dist = []
    total_rallies = []
    
    for player_name, scores in sorted(results.items()):
        if scores['num_rallies'] > 0:
            print(f"\n{player_name}:")
            print(f"  Land DTW:  {scores['land_dtw']:.4f}")
            print(f"  Move DTW:  {scores['move_dtw']:.4f}")
            print(f"  Edit Dist: {scores['edit_dist']:.4f}")
            print(f"  Rallies:   {scores['num_rallies']}")
            print(f"  Shots:     {scores['num_shots']}")
            
            total_land_dtw.append(scores['land_dtw'])
            total_move_dtw.append(scores['move_dtw'])
            total_edit_dist.append(scores['edit_dist'])
            total_rallies.append(scores['num_rallies'])
    
    if total_rallies:
        print("\n" + "-"*70)
        print(f"整體平均:")
        print(f"  Land DTW:  {sum(total_land_dtw)/len(total_land_dtw):.4f}")
        print(f"  Move DTW:  {sum(total_move_dtw)/len(total_move_dtw):.4f}")
        print(f"  Edit Dist: {sum(total_edit_dist)/len(total_edit_dist):.4f}")
        print(f"  總 Rallies: {sum(total_rallies)}")
    
    print("="*70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='評估 BC 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 在相同實驗類型上評估
  python Scripts/eval_bc.py --experiment norm
  python Scripts/eval_bc.py --experiment shifted
  
  # 跨實驗評估（norm 模型在 shifted 上測試）
  python Scripts/eval_bc.py --train-on norm --test-on shifted
  
  # 使用自訂模型
  python Scripts/eval_bc.py --experiment shifted --model-name my_model.pth
  
  # 查看可用的模型和結果
  python Scripts/eval_bc.py --list
  
  # 查看評估結果
  python Scripts/eval_bc.py --view
  python Scripts/eval_bc.py --view --result-name shifted_eval.json
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['norm', 'shifted'],
        help='實驗類型（train 和 test 使用相同類型）'
    )
    parser.add_argument(
        '--train-on',
        type=str,
        choices=['norm', 'shifted'],
        help='模型訓練時使用的實驗類型（預設與 --experiment 相同）'
    )
    parser.add_argument(
        '--test-on',
        type=str,
        choices=['norm', 'shifted'],
        help='要測試的實驗類型（預設與 --experiment 相同）'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='自訂模型檔名（在 weight 目錄中）'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='運算裝置（預設使用 config 設定）'
    )
    parser.add_argument(
        '--skip-first-n',
        type=int,
        help='跳過前 n 個 shots（預設使用 config 設定）'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出可用的模型和評估結果'
    )
    parser.add_argument(
        '--view',
        action='store_true',
        help='查看最新的評估結果'
    )
    parser.add_argument(
        '--result-name',
        type=str,
        help='指定要查看的結果檔名（配合 --view 使用）'
    )
    
    args = parser.parse_args()
    
    # 如果只列出資訊
    if args.list:
        list_available_models()
        return 0
    
    # 如果查看結果
    if args.view:
        success = view_result(args.result_name)
        return 0 if success else 1
    
    # 檢查必要參數
    if not args.experiment and (not args.train_on or not args.test_on):
        parser.print_help()
        print("\n❌ 錯誤: 必須指定 --experiment 或同時指定 --train-on 和 --test-on")
        return 1
    
    # 決定 train 和 test 類型
    train_exp = args.train_on or args.experiment
    test_exp = args.test_on or args.experiment
    
    # 執行評估
    success = eval_bc(
        train_experiment=train_exp,
        test_experiment=test_exp,
        model_name=args.model_name,
        device=args.device,
        skip_first_n=args.skip_first_n
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
