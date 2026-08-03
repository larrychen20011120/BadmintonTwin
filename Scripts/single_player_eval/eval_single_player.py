#!/usr/bin/env python3
"""
單一選手快速評估工具

這個腳本可以針對單一選手進行快速評估，適合用於開發測試或快速驗證演算法效果。

使用方式:
    # 評估單一選手的BC模型
    python Scripts/single_player_eval/eval_single_player.py \
        --algorithm bc \
        --player "Viktor AXELSEN"

    # 評估單一選手的RallyNetV2
    python Scripts/single_player_eval/eval_single_player.py \
        --algorithm rallynetv2 \
        --player "Kento MOMOTA"

    # 列出所有可用的選手
    python Scripts/single_player_eval/eval_single_player.py --list-players

    # 列出所有支援的演算法
    python Scripts/single_player_eval/eval_single_player.py --list-algorithms

支援的演算法:
    - bc: Behavior Cloning
    - rallynet: RallyNet
    - rallynetv2: RallyNetV2
    - naive_rag: NaiveRAG
    - random: Random Agent
"""

import argparse
import os
import sys
import json

# 加入專案路徑
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_path)

from Utils.llm import LLM
from Utils.runtime_report import algorithm_runtime_rows, llm_rows, print_runtime_box


def get_algorithm_map():
    """
    延遲載入演算法映射
    避免在只列出資訊時載入不必要的相依性
    """
    from Scripts.single_player_eval.evaluators import (
        BCEvaluator,
        RallyNetEvaluator,
        RallyNetV2Evaluator,
        NaiveRAGEvaluator,
        TimeRAGEvaluator,
        TwinEvaluator,
        RandomEvaluator,
        ShuttleNetEvaluator
    )
    
    return {
        'bc': BCEvaluator,
        'rallynet': RallyNetEvaluator,
        'rallynetv2': RallyNetV2Evaluator,
        'naive_rag': NaiveRAGEvaluator,
        'timerag': TimeRAGEvaluator,
        'twin': TwinEvaluator,
        'random': RandomEvaluator,
        'shuttlenet': ShuttleNetEvaluator,
    }


def load_config():
    """載入設定檔"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def list_available_players(experiment='norm'):
    """列出所有可用的選手"""
    config = load_config()
    test_split_path = config['paths'][f'{experiment}_split']
    
    with open(test_split_path, 'r') as f:
        test_split = json.load(f)
    
    players = sorted(test_split.get('test', {}).keys())
    
    print(f"\n可用的選手 (Experiment: {experiment})")
    print("="*70)
    
    # 顯示選手及其match數量
    for player in players:
        match_count = len(test_split['test'][player])
        print(f"  - {player:<35} ({match_count} matches)")
    
    print("="*70)
    print(f"總計: {len(players)} 位選手\n")
    
    return players


def list_available_algorithms():
    """列出所有支援的演算法"""
    print("\n支援的演算法")
    print("="*70)

    algorithms = [
        ('bc', 'Behavior Cloning - 行為複製模型', True),
        ('rallynet', 'RallyNet - two-stage SDE-based model', True),
        ('rallynetv2', 'RallyNetV2 - 擴展版RallyNet', True),
        ('naive_rag', 'NaiveRAG - 基於檢索的LLM方法', True),
        ('timerag', 'TimeRAG - 時間窗口檢索的LLM方法', True),
        ('twin', 'Twin - aspect-memory retrieval 的LLM方法', True),
        ('random', 'Random Agent - 隨機基線', True),
        ('shuttlenet', 'ShuttleNet - position-aware rally/player fusion', True),
    ]

    for algo, desc, implemented in algorithms:
        status = "✓" if implemented else "✗"
        print(f"  {status} {algo:<15} {desc}")

    print("="*70)
    print()


def output_experiment_label(experiment, style_shift_instruction_mode):
    """Return the storage label; split/data selection still uses experiment."""
    if experiment != 'shifted':
        return experiment
    if style_shift_instruction_mode == 'auto':
        return 'shifted_with_instruction'
    return 'shifted_without_instruction'


def create_evaluator(algorithm, player_name, experiment='norm', device='cuda', 
                     skip_first_n=0, **kwargs):
    """
    建立對應的Evaluator實例
    
    Args:
        algorithm: 演算法名稱
        player_name: 選手名稱
        experiment: 實驗類型
        device: 運算裝置
        skip_first_n: 跳過前n個shots
        **kwargs: 額外參數
    
    Returns:
        BaseEvaluator: Evaluator實例
    """
    algorithm_map = get_algorithm_map()
    
    if algorithm not in algorithm_map:
        raise ValueError(
            f"不支援的演算法: {algorithm}\n"
            f"支援的演算法: {list(algorithm_map.keys())}"
        )
    
    evaluator_class = algorithm_map[algorithm]
    if algorithm in ('bc', 'rallynet', 'shuttlenet'):
        experiment = 'rally'
    
    if evaluator_class is None:
        raise NotImplementedError(
            f"演算法 '{algorithm}' 尚未實現單一選手評估功能"
        )
    
    return evaluator_class(
        player_name=player_name,
        experiment=experiment,
        device=device,
        skip_first_n=skip_first_n,
        **kwargs
    )


def evaluate_single_player(algorithm, player_name, experiment='norm', device='cuda',
                           skip_first_n=0, save_results=True, **kwargs):
    """
    評估單一選手
    
    Args:
        algorithm: 演算法名稱
        player_name: 選手名稱
        experiment: 實驗類型
        device: 運算裝置
        skip_first_n: 跳過前n個shots
        save_results: 是否儲存結果
        **kwargs: 額外參數傳遞給Evaluator
    
    Returns:
        dict: 評估結果
    """
    print("="*80)
    print("單一選手快速評估")
    print("="*80)
    print(f"演算法: {algorithm}")
    print(f"選手: {player_name}")
    print(f"實驗類型: {experiment}")
    print(f"裝置: {device}")
    print(f"Skip First N: {skip_first_n}")
    print("="*80)
    print()
    
    # 建立Evaluator
    try:
        evaluator = create_evaluator(
            algorithm=algorithm,
            player_name=player_name,
            experiment=experiment,
            device=device,
            skip_first_n=skip_first_n,
            **kwargs
        )
    except Exception as e:
        print(f"❌ 建立Evaluator失敗: {e}")
        return None
    
    # 執行評估
    try:
        print(f"開始評估 {player_name}...")
        results = evaluator.simulate()
        
        print(f"✅ 模擬完成，共收集 {len(results)} 個rallies")
        
        return results
        
    except Exception as e:
        print(f"❌ 評估過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='單一選手快速評估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 評估單一選手的BC模型
  python Scripts/single_player_eval/eval_single_player.py \\
      --algorithm bc \\
      --player "Viktor AXELSEN"

  # 評估RallyNetV2
  python Scripts/single_player_eval/eval_single_player.py \\
      --algorithm rallynetv2 \\
      --player "Kento MOMOTA"

  # 評估NaiveRAG
  python Scripts/single_player_eval/eval_single_player.py \\
      --algorithm naive_rag \\
      --player "Akane YAMAGUCHI"

  # 評估Random Agent
  python Scripts/single_player_eval/eval_single_player.py \\
      --algorithm random \\
      --player "An Se Young" \\
      --seed 42

  # 列出所有選手
  python Scripts/single_player_eval/eval_single_player.py --list-players

  # 列出所有演算法
  python Scripts/single_player_eval/eval_single_player.py --list-algorithms
        """
    )
    
    # 主要參數
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['bc', 'rallynet', 'rallynetv2', 'naive_rag', 'timerag', 'twin', 'random', 'shuttlenet'],
        help='要使用的演算法'
    )
    
    parser.add_argument(
        '--player',
        type=str,
        help='選手完整名稱（用雙引號包起來）'
    )
    
    # 可選參數
    parser.add_argument(
        '--experiment',
        type=str,
        default='norm',
        choices=['norm', 'shifted'],
        help='實驗類型（預設: norm）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='運算裝置（預設讀取 config evaluation.device）'
    )
    
    parser.add_argument(
        '--skip-first-n',
        type=int,
        default=None,
        help='跳過前n個shots（預設讀取 config）'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不儲存評估結果'
    )
    
    parser.add_argument(
        '--save-trajectories',
        action='store_true',
        help='將每次 rally simulation 儲存為獨立 JSON 檔案'
    )
    
    # 資訊參數
    parser.add_argument(
        '--list-players',
        action='store_true',
        help='列出所有可用的選手'
    )
    
    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help='列出所有支援的演算法'
    )
    
    # 演算法特定參數
    parser.add_argument(
        '--model-path',
        type=str,
        help='自訂模型路徑（適用於bc, rallynet, rallynetv2, shuttlenet）'
    )

    parser.add_argument(
        '--distribution',
        type=str,
        default=None,
        choices=['deterministic', 'gaussian', 'gmm'],
        help='BC輸出模式（僅適用於bc）：deterministic、gaussian 或 gmm（預設讀取 config）'
    )
    parser.add_argument(
        '--gmm-num',
        type=int,
        help='GMM component 數量（僅適用於 bc gmm，預設讀取 config）'
    )
    
    parser.add_argument(
        '--hyperparameters-path',
        type=str,
        help='自訂hyperparameters路徑（僅適用於rallynetv2）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='隨機種子（僅適用於random）'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Classic RAG檢索數量（僅適用於naive_rag, timerag, twin, twin2，預設讀取 config retrieval.classic_top_k）'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=None,
        help='時間窗口大小（僅適用於timerag，預設讀取 config）'
    )
    
    parser.add_argument(
        '--max-scale',
        type=int,
        default=None,
        help='Deprecated; Twin now uses --lambda-weight.'
    )

    parser.add_argument(
        '--lambda-weight',
        type=float,
        default=None,
        help='Twin numeric/semantic retrieval mixing weight（僅適用於twin，預設讀取 config）'
    )

    parser.add_argument(
        '--style-shift-instruction',
        type=str,
        default='auto',
        choices=['auto', 'off'],
        help=('Style-shift instruction 模式（僅適用於 legacy shifted evaluators）：'
              'auto 使用 shifted prompt 並注入 instruction；'
              'off 使用一般 prompt 且不注入 instruction')
    )

    parser.add_argument(
        '--verbose-style-shift',
        action='store_true',
        help='印出 style-shift instruction 是否套用、套用到哪位選手與內容'
    )
    
    args = parser.parse_args()
    
    # 如果只列出資訊
    if args.list_players:
        list_available_players(args.experiment)
        return 0
    
    if args.list_algorithms:
        list_available_algorithms()
        return 0
    
    # 檢查必要參數
    if not args.algorithm or not args.player:
        parser.print_help()
        print("\n❌ 錯誤: 必須指定 --algorithm 和 --player")
        print("   或使用 --list-players 查看可用的選手")
        return 1

    if args.algorithm in ('naive_rag', 'timerag', 'twin'):
        print(
            "NaiveRAG/TimeRAG/Twin must evaluate complete physical rallies with "
            "Scripts/single_player_eval/eval_all_rallies.py."
        )
        return 1

    if args.algorithm in ('bc', 'rallynet', 'shuttlenet'):
        args.experiment = 'rally'

    config = load_config()
    eval_config = config.get('evaluation', {})
    algorithm_config = config.get(args.algorithm, {})
    retrieval_config = config.get('retrieval', {})
    device = args.device or eval_config.get('device', 'cuda')
    skip_first_n = (
        args.skip_first_n
        if args.skip_first_n is not None
        else algorithm_config.get('skip_first_n', eval_config.get('skip_first_n', 0))
    )
    effective_top_k = args.top_k if args.top_k is not None else retrieval_config.get('classic_top_k', retrieval_config.get('top_k', algorithm_config.get('top_k')))
    effective_window_size = args.window_size if args.window_size is not None else algorithm_config.get('window_size')
    if args.algorithm == 'twin' and args.max_scale is not None:
        print("Warning: --max-scale is deprecated and ignored for Twin; use --lambda-weight.")
    effective_lambda_weight = args.lambda_weight if args.lambda_weight is not None else algorithm_config.get('lambda_weight')
    
    # 收集額外參數
    kwargs = {}
    if args.model_path:
        kwargs['model_path'] = args.model_path
    if args.algorithm == 'bc' and args.distribution:
        kwargs['distribution'] = args.distribution
    if args.algorithm == 'bc' and args.gmm_num:
        kwargs['gmm_num'] = args.gmm_num
    if args.hyperparameters_path and args.algorithm == 'rallynetv2':
        kwargs['hyperparameters_path'] = args.hyperparameters_path
    if args.seed is not None:
        kwargs['seed'] = args.seed
    if args.algorithm in ('naive_rag', 'timerag', 'twin') and args.top_k is not None:
        kwargs['top_k'] = args.top_k
    if args.algorithm == 'timerag' and args.window_size is not None:
        kwargs['window_size'] = args.window_size
    if args.algorithm == 'twin' and effective_lambda_weight is not None:
        kwargs['lambda_weight'] = effective_lambda_weight
    if args.save_trajectories:
        kwargs['save_trajectories'] = True
    if args.algorithm in ('naive_rag', 'timerag', 'twin'):
        kwargs['style_shift_instruction_mode'] = args.style_shift_instruction
        kwargs['verbose_style_shift'] = args.verbose_style_shift
        kwargs['output_experiment'] = output_experiment_label(args.experiment, args.style_shift_instruction)

    rows = algorithm_runtime_rows(
        config,
        args.algorithm,
        experiment=args.experiment,
        player=args.player,
        device=device,
        skip_first_n=skip_first_n,
        top_k=effective_top_k,
        retrieval_mode=retrieval_config.get('mode'),
        normal_top_k=retrieval_config.get('normal_top_k'),
        miss_top_k=retrieval_config.get('miss_top_k'),
        window_size=effective_window_size,
        lambda_weight=effective_lambda_weight,
    )
    if args.algorithm in ('naive_rag', 'timerag', 'twin'):
        rows.extend(llm_rows(LLM.resolve_effective_settings()))
    print_runtime_box(f"Runtime Parameters: eval_single_player/{args.algorithm}", rows)
     
    # 執行評估
    results = evaluate_single_player(
        algorithm=args.algorithm,
        player_name=args.player,
        experiment=args.experiment,
        device=device,
        skip_first_n=skip_first_n,
        save_results=not args.no_save,
        **kwargs
    )
    
    return 0 if results is not None else 1


if __name__ == "__main__":
    exit(main())
