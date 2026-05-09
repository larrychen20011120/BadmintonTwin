#!/usr/bin/env python3
"""
統一資料準備腳本
準備 BC、NaiveRAG、TimeRAG 和 Twin 的 norm 與 shifted 資料

使用方式:
    python Scripts/prepare_all_data.py              # 準備所有資料
    python Scripts/prepare_all_data.py --norm       # 只準備 norm 資料
    python Scripts/prepare_all_data.py --shifted    # 只準備 shifted 資料
    python Scripts/prepare_all_data.py --bc         # 只準備 BC 資料
    python Scripts/prepare_all_data.py --naive-rag  # 只準備 NaiveRAG 資料
    python Scripts/prepare_all_data.py --timerag    # 只準備 TimeRAG 資料
    python Scripts/prepare_all_data.py --twin       # 只準備 Twin 資料
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


def run_baseline_prepare(
    train_split,
    bc_output,
    naive_rag_output,
    timerag_output=None,
    twin_output=None,
    prepare_bc=True,
    prepare_rag=True,
    prepare_timerag=False,
    prepare_twin=False
):
    """
    執行 baseline_prepare.py

    Args:
        train_split: train/test split 檔案路徑
        bc_output: BC 資料輸出目錄
        naive_rag_output: NaiveRAG 資料輸出目錄
        timerag_output: TimeRAG 資料輸出目錄
        twin_output: Twin 資料輸出目錄
        prepare_bc: 是否準備 BC 資料
        prepare_rag: 是否準備 NaiveRAG 資料
        prepare_timerag: 是否準備 TimeRAG 資料
        prepare_twin: 是否準備 Twin 資料
    """
    config = load_config()
    merged_csv = config['paths']['merged_csv']
    player_map = config['paths']['player_map']
    
    script_path = os.path.join(project_path, 'Preprocessing', 'baseline_prepare.py')
    
    if prepare_bc:
        print(f"\n{'='*70}")
        print(f"準備 BC 資料")
        print(f"  Split: {train_split}")
        print(f"  輸出: {bc_output}")
        print(f"{'='*70}")
        
        cmd = [
            'python', script_path,
            '--bc',
            '--train_split', train_split,
            '--merged_csv', merged_csv,
            '--player_map', player_map,
            '--bc_output', bc_output
        ]
        
        result = subprocess.run(cmd, cwd=project_path)
        if result.returncode != 0:
            print(f"❌ BC 資料準備失敗: {train_split}")
            return False
        print(f"✅ BC 資料準備完成: {bc_output}")
    
    if prepare_rag:
        print(f"\n{'='*70}")
        print(f"準備 NaiveRAG 資料")
        print(f"  Split: {train_split}")
        print(f"  輸出: {naive_rag_output}")
        print(f"{'='*70}")
        
        cmd = [
            'python', script_path,
            '--naive-rag',
            '--train_split', train_split,
            '--merged_csv', merged_csv,
            '--player_map', player_map,
            '--naive_rag_output', naive_rag_output
        ]
        
        result = subprocess.run(cmd, cwd=project_path)
        if result.returncode != 0:
            print(f"❌ NaiveRAG 資料準備失敗: {train_split}")
            return False
        print(f"✅ NaiveRAG 資料準備完成: {naive_rag_output}")
    
    if prepare_timerag and timerag_output:
        print(f"\n{'='*70}")
        print(f"準備 TimeRAG 資料")
        print(f"  Split: {train_split}")
        print(f"  輸出: {timerag_output}")
        print(f"{'='*70}")

        cmd = [
            'python', script_path,
            '--timerag',
            '--train_split', train_split,
            '--merged_csv', merged_csv,
            '--player_map', player_map,
            '--timerag_output', timerag_output
        ]

        result = subprocess.run(cmd, cwd=project_path)
        if result.returncode != 0:
            print(f"❌ TimeRAG 資料準備失敗: {train_split}")
            return False
        print(f"✅ TimeRAG 資料準備完成: {timerag_output}")

    if prepare_twin and twin_output:
        print(f"\n{'='*70}")
        print(f"準備 Twin (I-MSDP) 資料")
        print(f"  Split: {train_split}")
        print(f"  輸出: {twin_output}")
        print(f"{'='*70}")

        cmd = [
            'python', script_path,
            '--twin',
            '--train_split', train_split,
            '--merged_csv', merged_csv,
            '--player_map', player_map,
            '--twin_output', twin_output
        ]

        result = subprocess.run(cmd, cwd=project_path)
        if result.returncode != 0:
            print(f"❌ Twin 資料準備失敗: {train_split}")
            return False
        print(f"✅ Twin 資料準備完成: {twin_output}")

    return True


def create_style_shift_split_if_needed(config):
    """如果 shifted_split 不存在，則創建它"""
    shifted_split_path = config['paths']['shifted_split']
    
    if not os.path.exists(shifted_split_path):
        print(f"\n⚠ {shifted_split_path} 不存在，正在創建...")
        
        script_path = os.path.join(os.path.dirname(__file__), 'create_style_shift_split.py')
        result = subprocess.run(['python', script_path], cwd=project_path)
        
        if result.returncode != 0:
            print("❌ 創建 style_shift_split.json 失敗")
            return False
    else:
        print(f"\n✅ style_shift_split.json 已存在: {shifted_split_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='準備 BC、NaiveRAG、TimeRAG 和 Twin 的訓練資料',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python Scripts/prepare_all_data.py              # 準備所有資料 (norm + shifted)
  python Scripts/prepare_all_data.py --norm       # 只準備 norm 資料
  python Scripts/prepare_all_data.py --shifted    # 只準備 shifted 資料
  python Scripts/prepare_all_data.py --bc         # 只準備 BC 資料 (兩種 split)
  python Scripts/prepare_all_data.py --naive-rag  # 只準備 NaiveRAG 資料 (兩種 split)
  python Scripts/prepare_all_data.py --timerag    # 只準備 TimeRAG 資料 (兩種 split)
  python Scripts/prepare_all_data.py --twin       # 只準備 Twin 資料 (兩種 split)
        """
    )
    
    parser.add_argument(
        '--norm',
        action='store_true',
        help='只準備 norm (train_test_split) 資料'
    )
    parser.add_argument(
        '--shifted',
        action='store_true',
        help='只準備 shifted (style_shift) 資料'
    )
    parser.add_argument(
        '--bc',
        action='store_true',
        help='只準備 BC 資料'
    )
    parser.add_argument(
        '--naive-rag',
        action='store_true',
        help='只準備 NaiveRAG 資料'
    )
    parser.add_argument(
        '--timerag',
        action='store_true',
        help='只準備 TimeRAG 資料'
    )
    parser.add_argument(
        '--twin',
        action='store_true',
        help='只準備 Twin (I-MSDP) 資料'
    )

    args = parser.parse_args()
    
    # 載入設定檔
    config = load_config()
    
    # 如果沒有指定特定類型，則全部準備
    specific_baseline = args.bc or args.naive_rag or args.timerag or args.twin
    specific_experiment = args.norm or args.shifted

    prepare_norm = args.norm or (not specific_experiment and not specific_baseline)
    prepare_shifted = args.shifted or (not specific_experiment and not specific_baseline)
    prepare_bc = args.bc or (not args.naive_rag and not args.timerag and not args.twin)
    prepare_rag = args.naive_rag or (not args.bc and not args.timerag and not args.twin)
    prepare_timerag_data = args.timerag or (not args.bc and not args.naive_rag and not args.twin)
    prepare_twin_data = args.twin or (not args.bc and not args.naive_rag and not args.timerag)

    # 如果指定了特定類型，則按照指定來
    if specific_experiment:
        prepare_norm = args.norm
        prepare_shifted = args.shifted
    
    print("="*70)
    print("BadmintonTwin 資料準備")
    print("="*70)
    print(f"準備項目:")
    print(f"  - Norm 資料:     {'是' if prepare_norm else '否'}")
    print(f"  - Shifted 資料:  {'是' if prepare_shifted else '否'}")
    print(f"  - BC 資料:       {'是' if prepare_bc else '否'}")
    print(f"  - NaiveRAG 資料: {'是' if prepare_rag else '否'}")
    print(f"  - TimeRAG 資料:  {'是' if prepare_timerag_data else '否'}")
    print(f"  - Twin 資料:     {'是' if prepare_twin_data else '否'}")
    print("="*70)
    
    # 準備 shifted 資料前，先確認 style_shift_split.json 存在
    if prepare_shifted:
        if not create_style_shift_split_if_needed(config):
            return 1
    
    success = True
    
    # 準備 Norm 資料
    if prepare_norm:
        if prepare_bc or prepare_rag or prepare_timerag_data or prepare_twin_data:
            result = run_baseline_prepare(
                train_split=config['paths']['norm_split'],
                bc_output=config['paths']['bc']['data']['norm'],
                naive_rag_output=config['paths']['naive_rag']['data']['norm'],
                timerag_output=config['paths']['timerag']['data']['norm'] if 'timerag' in config['paths'] else None,
                twin_output=config['paths']['twin']['data']['norm'] if 'twin' in config['paths'] else None,
                prepare_bc=prepare_bc,
                prepare_rag=prepare_rag,
                prepare_timerag=prepare_timerag_data,
                prepare_twin=prepare_twin_data
            )
            if not result:
                success = False

    # 準備 Shifted 資料
    if prepare_shifted:
        if prepare_bc or prepare_rag or prepare_timerag_data or prepare_twin_data:
            result = run_baseline_prepare(
                train_split=config['paths']['shifted_split'],
                bc_output=config['paths']['bc']['data']['shifted'],
                naive_rag_output=config['paths']['naive_rag']['data']['shifted'],
                timerag_output=config['paths']['timerag']['data']['shifted'] if 'timerag' in config['paths'] else None,
                twin_output=config['paths']['twin']['data']['shifted'] if 'twin' in config['paths'] else None,
                prepare_bc=prepare_bc,
                prepare_rag=prepare_rag,
                prepare_timerag=prepare_timerag_data,
                prepare_twin=prepare_twin_data
            )
            if not result:
                success = False
    
    # 輸出總結
    print("\n" + "="*70)
    print("資料準備完成!")
    print("="*70)
    
    if prepare_bc:
        print("\nBC 資料位置:")
        if prepare_norm:
            print(f"  - Norm:    {config['paths']['bc']['data']['norm']}")
        if prepare_shifted:
            print(f"  - Shifted: {config['paths']['bc']['data']['shifted']}")
    
    if prepare_rag:
        print("\nNaiveRAG 資料位置:")
        if prepare_norm:
            print(f"  - Norm:    {config['paths']['naive_rag']['data']['norm']}")
        if prepare_shifted:
            print(f"  - Shifted: {config['paths']['naive_rag']['data']['shifted']}")
    
    if prepare_timerag_data and 'timerag' in config['paths']:
        print("\nTimeRAG 資料位置:")
        if prepare_norm:
            print(f"  - Norm:    {config['paths']['timerag']['data']['norm']}")
        if prepare_shifted:
            print(f"  - Shifted: {config['paths']['timerag']['data']['shifted']}")

    if prepare_twin_data and 'twin' in config['paths']:
        print("\nTwin (I-MSDP) 資料位置:")
        if prepare_norm:
            print(f"  - Norm:    {config['paths']['twin']['data']['norm']}")
        if prepare_shifted:
            print(f"  - Shifted: {config['paths']['twin']['data']['shifted']}")

    print("="*70)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
