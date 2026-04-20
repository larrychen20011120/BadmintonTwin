#!/usr/bin/env python3
"""
Random Agent 評估腳本
評估 Random Agent 在 norm 或 shifted test 資料上的表現

使用方式:
    python Scripts/eval_random.py --experiment norm       # 在 norm test 上評估
    python Scripts/eval_random.py --experiment shifted    # 在 shifted test 上評估
    python Scripts/eval_random.py --experiment norm --seed 123  # 指定隨機種子
"""

import argparse
import os
import sys
import subprocess
import yaml
import json

# 加入專案路徑
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


def load_config():
    """載入設定檔"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def eval_random(test_experiment, seed=None, skip_first_n=None):
    """
    評估 Random Agent
    
    Args:
        test_experiment: 要測試的實驗類型 ('norm' 或 'shifted')
        seed: 隨機種子（可選）
        skip_first_n: 跳過前 n 個 shots（可選）
    """
    config = load_config()
    
    # 取得路徑
    results_dir = config['paths']['random']['results_dir']
    
    # 決定 test split
    test_split = config['paths'][f'{test_experiment}_split']
    
    # 決定輸出路徑
    output_name = f"{test_experiment}_eval.json"
    output_path = os.path.join(results_dir, output_name)
    
    # 取得評估參數
    skip_first_n = skip_first_n if skip_first_n is not None else config['evaluation']['skip_first_n']
    
    # 檢查必要檔案
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
    print(f"Random Agent 評估 - {test_experiment.upper()}")
    print("="*70)
    print(f"測試資料: {test_split}")
    print(f"輸出結果: {output_path}")
    print(f"評估參數:")
    print(f"  - Seed: {seed if seed is not None else 'None (random)'}")
    print(f"  - Skip First N: {skip_first_n}")
    print("="*70)
    
    # 執行評估
    random_script = os.path.join(project_path, 'Agent', 'random_agent.py')
    
    cmd = [
        'python', random_script,
        '--test_split', test_split,
        '--merged_csv', config['paths']['merged_csv'],
        '--player_map', config['paths']['player_map'],
        '--output', output_path
    ]
    
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    
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


def list_available_results():
    """列出可用的評估結果"""
    config = load_config()
    results_dir = config['paths']['random']['results_dir']
    
    print("\nRandom Agent 評估結果:")
    print("-"*70)
    
    if os.path.exists(results_dir):
        results = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if results:
            for result_file in sorted(results):
                result_path = os.path.join(results_dir, result_file)
                size_kb = os.path.getsize(result_path) / 1024
                print(f"  ✓ {result_file} ({size_kb:.2f} KB)")
        else:
            print("  (無結果)")
    else:
        print(f"  (目錄不存在)")
    
    print("-"*70)


def view_result(result_name=None):
    """查看評估結果"""
    config = load_config()
    results_dir = config['paths']['random']['results_dir']
    
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
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n評估結果: {os.path.basename(result_path)}")
    print("="*70)
    
    # 計算平均
    total_land_dtw = []
    total_move_dtw = []
    total_shot_ctc = []
    total_edit_dist = []
    total_rallies = []
    
    for player_name, scores in sorted(results.items()):
        if scores['num_rallies'] > 0:
            print(f"\n{player_name}:")
            print(f"  Land DTW:  {scores['land_dtw']:.4f}")
            print(f"  Move DTW:  {scores['move_dtw']:.4f}")
            print(f"  Shot CTC:  {scores['shot_ctc']:.4f}")
            print(f"  Edit Dist: {scores['edit_dist']:.4f}")
            print(f"  Rallies:   {scores['num_rallies']}")
            print(f"  Shots:     {scores['num_shots']}")
            
            total_land_dtw.append(scores['land_dtw'])
            total_move_dtw.append(scores['move_dtw'])
            total_shot_ctc.append(scores['shot_ctc'])
            total_edit_dist.append(scores['edit_dist'])
            total_rallies.append(scores['num_rallies'])
    
    if total_rallies:
        print("\n" + "-"*70)
        print(f"整體平均:")
        print(f"  Land DTW:  {sum(total_land_dtw)/len(total_land_dtw):.4f}")
        print(f"  Move DTW:  {sum(total_move_dtw)/len(total_move_dtw):.4f}")
        print(f"  Shot CTC:  {sum(total_shot_ctc)/len(total_shot_ctc):.4f}")
        print(f"  Edit Dist: {sum(total_edit_dist)/len(total_edit_dist):.4f}")
        print(f"  總 Rallies: {sum(total_rallies)}")
    
    print("="*70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='評估 Random Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 在 norm test 上評估
  python Scripts/eval_random.py --experiment norm
  
  # 在 shifted test 上評估
  python Scripts/eval_random.py --experiment shifted
  
  # 指定隨機種子
  python Scripts/eval_random.py --experiment norm --seed 123
  
  # 跳過前 3 個 shots
  python Scripts/eval_random.py --experiment norm --skip-first-n 3
  
  # 查看可用的結果
  python Scripts/eval_random.py --list
  
  # 查看評估結果
  python Scripts/eval_random.py --view
  python Scripts/eval_random.py --view --result-name shifted_eval.json
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['norm', 'shifted'],
        help='測試的實驗類型'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='隨機種子（預設: None）'
    )
    parser.add_argument(
        '--skip-first-n',
        type=int,
        help='跳過前 n 個 shots（預設使用 config 設定）'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出可用的評估結果'
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
        list_available_results()
        return 0
    
    # 如果查看結果
    if args.view:
        success = view_result(args.result_name)
        return 0 if success else 1
    
    # 檢查必要參數
    if not args.experiment:
        parser.print_help()
        print("\n❌ 錯誤: 必須指定 --experiment")
        return 1
    
    # 執行評估
    success = eval_random(
        test_experiment=args.experiment,
        seed=args.seed,
        skip_first_n=args.skip_first_n
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
