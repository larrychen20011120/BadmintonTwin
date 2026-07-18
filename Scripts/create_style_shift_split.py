#!/usr/bin/env python3
"""
將 style_shift.json 轉換為 train_test_split 格式
- normal -> train
- shifted -> test

使用方式:
    python Scripts/create_style_shift_split.py
    python Scripts/create_style_shift_split.py --output custom_split.json
"""

import json
import argparse
import os
import sys
import yaml

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Utils.runtime_report import print_runtime_box


def load_config():
    """載入設定檔"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_style_shift_split(config, output_path=None):
    """
    從 style_shift.json 創建 train/test split
    
    Args:
        config: 設定字典
        output_path: 輸出路徑（可選，預設使用 config 中的路徑）
    """
    input_path = config['paths']['style_shift_json']
    if output_path is None:
        output_path = config['paths']['shifted_split']
    
    print(f"讀取 style_shift.json: {input_path}")
    
    with open(input_path, 'r') as f:
        style_data = json.load(f)
    
    train_split = {}
    test_split = {}
    
    # 處理 normal 資料 -> train
    print("\n處理 normal 資料 (train)...")
    for player_data in style_data.get('normal', []):
        player_name = player_data['name']
        match_ids = player_data.get('match_ids', [])
        if match_ids:  # 只有包含比賽時才加入
            train_split[player_name] = match_ids
            print(f"  ✓ {player_name}: {len(match_ids)} 場比賽")
    
    # 處理 shifted 資料 -> test
    print("\n處理 shifted 資料 (test)...")
    skipped_players = []
    for player_data in style_data.get('shifted', []):
        player_name = player_data['name']
        match_ids = player_data.get('match_ids', [])
        
        # 過濾空 match_ids
        if match_ids and len(match_ids) > 0:
            test_split[player_name] = match_ids
            print(f"  ✓ {player_name}: {len(match_ids)} 場比賽")
        else:
            skipped_players.append(player_name)
    
    if skipped_players:
        print(f"\n  ⚠ 跳過以下玩家（無 shifted match_ids）: {', '.join(skipped_players)}")
    
    split_data = {
        'train': train_split,
        'test': test_split
    }
    
    # 儲存輸出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print("\n" + "="*60)
    print("Style Shift Split 創建完成!")
    print("="*60)
    print(f"輸出檔案: {output_path}")
    print(f"  - Train 玩家數: {len(train_split)}")
    print(f"  - Test 玩家數: {len(test_split)}")
    print(f"  - Train 總比賽數: {sum(len(v) for v in train_split.values())}")
    print(f"  - Test 總比賽數: {sum(len(v) for v in test_split.values())}")
    print("="*60)
    
    return split_data


def validate_split(config):
    """
    驗證兩種 split 檔案的存在性和內容
    """
    print("\n驗證 Split 檔案...")
    print("-"*60)
    
    splits = {
        'norm': config['paths']['norm_split'],
        'shifted': config['paths']['shifted_split']
    }
    
    for name, path in splits.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            train_count = len(data.get('train', {}))
            test_count = len(data.get('test', {}))
            print(f"✓ {name:10s}: {path}")
            print(f"           Train: {train_count} 玩家, Test: {test_count} 玩家")
        else:
            print(f"✗ {name:10s}: {path} (不存在)")
    
    print("-"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='從 style_shift.json 創建 train/test split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python Scripts/create_style_shift_split.py
  python Scripts/create_style_shift_split.py --validate
  python Scripts/create_style_shift_split.py --output custom_split.json
        """
    )
    parser.add_argument(
        '--output', 
        type=str,
        default=None,
        help='自訂輸出路徑（預設使用 config.yaml 中的設定）'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='只驗證現有的 split 檔案，不重新創建'
    )
    
    args = parser.parse_args()
    
    # 載入設定
    config = load_config()

    print_runtime_box(
        "Runtime Parameters: create_style_shift_split",
        [
            ("validate", args.validate),
            ("output", args.output or config['paths']['shifted_split']),
            ("style_shift_json", config['paths']['style_shift_json']),
        ],
    )
    
    if args.validate:
        validate_split(config)
    else:
        # 創建 style_shift split
        create_style_shift_split(config, args.output)
        print()
        # 驗證結果
        validate_split(config)
