#!/usr/bin/env python3
"""
通用 Prompt Tracer - 預覽 NaiveRAG、TimeRAG、Twin 的 Prompt 結構

使用方法:
    # Trace TimeRAG prompts
    python Scripts/single_player_eval/trace_prompts.py \
        --algorithm timerag \
        --player "Kento MOMOTA" \
        --window-size 3 \
        --num-examples 5 \
        --output-dir ./prompt_traces/

    # Trace Twin prompts  
    python Scripts/single_player_eval/trace_prompts.py \
        --algorithm twin \
        --player "Kento MOMOTA" \
        --max-scale 10 \
        --num-examples 5

    # Trace NaiveRAG prompts
    python Scripts/single_player_eval/trace_prompts.py \
        --algorithm naive_rag \
        --player "Viktor AXELSEN" \
        --top-k 5 \
        --num-examples 3

    # 輸出為 Markdown 格式（易於閱讀）
    python Scripts/single_player_eval/trace_prompts.py \
        --algorithm timerag \
        --player "Kento MOMOTA" \
        --format markdown \
        --output-dir ./prompt_traces/
"""

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

# 加入專案路徑
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_path)

import pandas as pd
import torch

from Utils.runtime_report import print_runtime_box

from Utils.helper import state_col, action_col


class PromptTracer(ABC):
    """Prompt Tracer 基類"""
    
    def __init__(self, player_name: str, experiment: str = 'norm', 
                 top_k: int = 5, num_examples: int = 5,
                 output_dir: str = './prompt_traces'):
        """
        Args:
            player_name: 選手名稱
            experiment: 實驗類型 ('norm' 或 'shifted')
            top_k: 檢索數量
            num_examples: 要生成的 example 數量
            output_dir: 輸出目錄
        """
        self.player_name = player_name
        self.experiment = experiment
        self.top_k = top_k
        self.num_examples = num_examples
        self.output_dir = output_dir
        
        # 載入設定
        import yaml
        config_path = os.path.join(project_path, 'Scripts', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 設定路徑
        self.merged_csv_path = self.config['paths']['merged_csv']
        self.player_map_path = self.config['paths']['player_map']
        self.test_split_path = self.config['paths'][f'{experiment}_split']
        
        # 載入資料
        self._load_data()
        
        # 建立輸出目錄
        self.algorithm_name = self.get_algorithm_name()
        self.output_path = os.path.join(
            output_dir, 
            f"{self.algorithm_name}_{player_name.replace(' ', '_')}"
        )
        os.makedirs(self.output_path, exist_ok=True)
        
    def _load_data(self):
        """載入資料和選手映射"""
        import json
        
        # 載入選手映射
        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)
        
        if self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")
        
        self.player_id = self.player2index[self.player_name]
        
        # 載入測試資料
        with open(self.test_split_path, 'r') as f:
            self.test_split = json.load(f)
        
        self.test_matches = self.test_split.get('test', {}).get(self.player_name, [])
        
        if not self.test_matches:
            raise ValueError(f"No test matches found for player '{self.player_name}'")
        
        # 載入 DataFrame
        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        
        print(f"✓ 載入資料完成: {len(self.df)} rows, {len(self.test_matches)} test matches")
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """返回算法名稱"""
        pass
    
    @abstractmethod
    def create_agent_components(self):
        """創建 Agent 組件（database, verbalizer 等）"""
        pass
    
    @abstractmethod
    def retrieve_and_build_context(self, state, info, launch: bool) -> Dict[str, Any]:
        """
        執行檢索並構建 context dict
        
        Returns:
            Dict 包含:
            - retrieved_items: 檢索結果列表
            - context_dict: 用於 Jinja2 模板的 context
            - state_dict: 當前狀態字典
        """
        pass
    
    def _state_to_env_format(self, state_18):
        """將 18-dim state 轉換為 env format"""
        opponent_type = int(state_18[12].item())
        player_pos = (state_18[5].item(), state_18[6].item())
        opponent_pos = (state_18[13].item(), state_18[14].item())
        landing_pos = (state_18[10].item(), state_18[11].item())
        return [opponent_type, player_pos, opponent_pos, landing_pos]
    
    def _build_info(self, row):
        """從 row 構建 info dict"""
        return {
            'rally': row['rally'],
            'round': [row['ball_round']],
            'score': [row['player_score'], row['opponent_score']],
            'action': [None]
        }
    
    def trace_prompts(self):
        """主要執行函數：收集並輸出 prompts"""
        print(f"\n{'='*80}")
        print(f"Tracing {self.algorithm_name.upper()} Prompts for {self.player_name}")
        print(f"{'='*80}\n")
        
        # 創建 Agent 組件
        self.create_agent_components()
        
        # 收集 examples
        examples = []
        example_count = 0
        
        for match_id in self.test_matches:
            if example_count >= self.num_examples:
                break
            
            match_df = self.df[self.df['match_id'] == match_id].copy()
            if len(match_df) == 0:
                continue
            
            # 找出對手
            players_in_match = match_df['player'].unique()
            opponent_name = None
            for p in players_in_match:
                if p != self.player_name:
                    opponent_name = p
                    break
            
            if opponent_name is None or opponent_name not in self.player2index:
                continue
            
            opponent_player_id = self.player2index[opponent_name]
            
            # 按 rally 分組
            rallies = match_df.groupby(['set', 'rally'])
            
            for (set_num, rally_num), rally_df in rallies:
                if example_count >= self.num_examples:
                    break
                
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                
                if len(rally_df) == 0:
                    continue
                
                # 重置歷史
                self.reset_history()
                
                # 收集這個 rally 的 history
                rally_history = []
                
                for shot_idx in range(len(rally_df)):
                    if example_count >= self.num_examples:
                        break
                    
                    row = rally_df.iloc[shot_idx]
                    ground_truth_player = row['player']
                    is_target_turn = (ground_truth_player == self.player_name)
                    
                    if not is_target_turn:
                        self.observe_history_row(row)
                        continue
                    
                    current_player_id = self.player_id if is_target_turn else opponent_player_id
                    
                    # 構建 state
                    state_values = [row[col] for col in state_col]
                    state_values[-1] = current_player_id
                    current_state = torch.tensor(state_values, dtype=torch.float32)
                    
                    info = self._build_info(row)
                    launch = (current_state[1].item() == 1)
                    
                    env_state = self._state_to_env_format(current_state)
                    
                    # 執行檢索並構建 context
                    result = self.retrieve_and_build_context(env_state, info, launch)
                    
                    # 渲染 user prompt
                    user_prompt = self.user_template.render(**result['context_dict'])
                    
                    # 記錄 example
                    example = {
                        'algorithm': self.algorithm_name,
                        'player': self.player_name,
                        'match_id': match_id,
                        'set': int(set_num),
                        'rally': int(rally_num),
                        'shot': shot_idx + 1,
                        'ball_round': int(row['ball_round']),
                        'is_launch': launch,
                        'system_prompt': self.system_prompt,
                        'user_prompt': user_prompt,
                        'retrieved_items': result['retrieved_items'],
                        'context_dict': result['context_dict'],
                        'state_dict': result['state_dict'],
                        'metadata': {
                            'player_score': float(row['player_score']),
                            'opponent_score': float(row['opponent_score']),
                            'opponent_type': int(row['opponent_type']),
                            'opponent_type_name': self._get_shot_name(int(row['opponent_type'])),
                            'player_location_area': self._get_area_name(
                                float(row['player_location_x']), 
                                float(row['player_location_y'])
                            ),
                            'hit_position_area': self._get_area_name(
                                float(row['hit_x']), 
                                float(row['hit_y'])
                            ),
                        }
                    }
                    
                    examples.append(example)
                    example_count += 1
                    
                    # 更新歷史
                    state_dict = self._build_state_dict(row, info)
                    rally_history.append(state_dict)
                    self.update_history(state_dict, info, launch)
                    
                    print(f"  ✓ Collected example {example_count}/{self.num_examples}: "
                          f"Match {match_id}, Rally {rally_num}, Shot {shot_idx + 1}")
        
        # 輸出結果
        self._save_results(examples)
        
        print(f"\n{'='*80}")
        print(f"Tracing Complete! {len(examples)} examples saved to:")
        print(f"  {self.output_path}")
        print(f"{'='*80}\n")
        
        return examples
    
    def _build_state_dict(self, row, info):
        """從 row 構建 state_dict"""
        from Utils.helper import coord2area
        from Utils.badminton_config import LANG_AREA_MAPPING, INDEX2EN_SHOT
        
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        
        # 確保所有值都轉換為 Python 原生類型
        def to_python(val):
            """將 numpy 類型轉換為 Python 原生類型"""
            if hasattr(val, 'item'):  # numpy 類型
                return val.item()
            return val
        
        state_dict = {
            'player': str(row['player']),
            'rally': int(to_python(info.get('rally', 0))),
            'ball_round': int(to_python(info['round'][-1])) if info.get('round') else 1,
            'player_score': float(to_python(player_score)),
            'opponent_score': float(to_python(opponent_score)),
            'score_status': int(to_python(player_score - opponent_score)),
            'player_location_x': float(to_python(row['player_location_x'])),
            'player_location_y': float(to_python(row['player_location_y'])),
            'player_location_area': self._get_area_name(
                float(to_python(row['player_location_x'])), 
                float(to_python(row['player_location_y']))
            ),
            'hit_x': float(to_python(row['hit_x'])),
            'hit_y': float(to_python(row['hit_y'])),
            'hit_position_area': self._get_area_name(
                float(to_python(row['hit_x'])), 
                float(to_python(row['hit_y']))
            ),
            'opponent_type': int(to_python(row['opponent_type'])),
            'opponent_type_name': self._get_shot_name(int(to_python(row['opponent_type']))),
            'opponent_location_x': float(to_python(row['opponent_location_x'])),
            'opponent_location_y': float(to_python(row['opponent_location_y'])),
            'opponent_location_area': self._get_area_name(
                float(to_python(row['opponent_location_x'])), 
                float(to_python(row['opponent_location_y']))
            ),
            'opponent_move_x': float(to_python(row.get('opponent_move_x', 0))),
            'opponent_move_y': float(to_python(row.get('opponent_move_y', 0))),
            'landing_x': float(to_python(row.get('landing_x', 0))),
            'landing_y': float(to_python(row.get('landing_y', 0))),
            'player_type': int(to_python(row.get('player_type', 0))),
            'moving_x': float(to_python(row.get('moving_x', 0))),
            'moving_y': float(to_python(row.get('moving_y', 0))),
        }

        if state_dict['player_type'] == 11:
            for key in ('landing_x', 'landing_y', 'moving_x', 'moving_y'):
                state_dict[key] = 0.0
        
        return state_dict
    
    def _get_area_name(self, x, y):
        """獲取區域名稱"""
        from Utils.helper import coord2area
        from Utils.badminton_config import LANG_AREA_MAPPING
        area_id = coord2area(float(x), float(y))
        return LANG_AREA_MAPPING.get(area_id, f'Unknown({area_id})')
    
    def _get_shot_name(self, shot_type):
        """獲取擊球類型名稱"""
        from Utils.badminton_config import INDEX2EN_SHOT
        return INDEX2EN_SHOT.get(int(shot_type), f'Unknown({shot_type})')
    
    @abstractmethod
    def reset_history(self):
        """重置歷史記錄"""
        pass
    
    @abstractmethod
    def update_history(self, state_dict, info, launch):
        """更新歷史記錄"""
        pass

    def observe_history_row(self, row):
        """Allow tracers that need both players to retain skipped rows."""
        return None
    
    def _save_results(self, examples: List[Dict]):
        """保存結果到檔案"""
        # 1. 保存 system prompt
        system_prompt_path = os.path.join(self.output_path, 'system_prompt.md')
        with open(system_prompt_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.algorithm_name.upper()} System Prompt\n\n")
            f.write(self.system_prompt)
        
        # 2. 保存每個 example（JSON 格式）
        for i, example in enumerate(examples, 1):
            # JSON 格式（完整資料）
            json_path = os.path.join(
                self.output_path, 
                f'example_{i:02d}_rally{example["rally"]:02d}_shot{example["shot"]:02d}.json'
            )
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(example, f, indent=2, ensure_ascii=False, default=str)
        
        # 3. 保存每個 example（Markdown 格式，易於閱讀）
        for i, example in enumerate(examples, 1):
            md_path = os.path.join(
                self.output_path,
                f'example_{i:02d}_rally{example["rally"]:02d}_shot{example["shot"]:02d}.md'
            )
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self._format_example_as_markdown(example, i))
        
        # 4. 保存 summary
        summary_path = os.path.join(self.output_path, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Prompt Trace Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Algorithm: {self.algorithm_name}\n")
            f.write(f"Player: {self.player_name}\n")
            f.write(f"Experiment: {self.experiment}\n")
            f.write(f"Top-K: {self.top_k}\n")
            f.write(f"Examples: {len(examples)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Files:\n")
            f.write(f"  - system_prompt.md\n")
            for i, example in enumerate(examples, 1):
                f.write(f"  - example_{i:02d}_rally{example['rally']:02d}_shot{example['shot']:02d}.json\n")
                f.write(f"  - example_{i:02d}_rally{example['rally']:02d}_shot{example['shot']:02d}.md\n")
        
        # 5. 保存完整的 prompts（所有 examples 合併，方便批量查看）
        all_prompts_path = os.path.join(self.output_path, 'all_prompts.md')
        with open(all_prompts_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.algorithm_name.upper()} Prompts - {self.player_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## System Prompt\n\n")
            f.write(f"```markdown\n{self.system_prompt}\n```\n\n")
            
            f.write(f"---\n\n")
            
            for i, example in enumerate(examples, 1):
                f.write(f"## Example {i}: Rally {example['rally']}, Shot {example['shot']}\n\n")
                f.write(f"**Match ID:** {example['match_id']}  \n")
                f.write(f"**Ball Round:** {example['ball_round']}  \n")
                f.write(f"**Is Launch:** {example['is_launch']}  \n")
                f.write(f"**Score:** {example['metadata']['player_score']}-{example['metadata']['opponent_score']}  \n")
                f.write(f"**Opponent Shot:** {example['metadata']['opponent_type_name']}  \n\n")
                
                f.write(f"### User Prompt\n\n")
                f.write(f"```markdown\n{example['user_prompt']}\n```\n\n")
                
                f.write(f"### Retrieved Items ({len(example['retrieved_items'])})\n\n")
                for item in example['retrieved_items']:
                    f.write(f"**Item {item.get('index', '?')}**  \n")
                    if 'distance' in item:
                        f.write(f"- Distance: {item['distance']:.4f}  \n")
                    if 'distance' in item:
                        f.write(f"- Distance: {item['distance']:.4f}  \n")
                    if 'similarity' in item:
                        f.write(f"- Similarity: {item['similarity']:.4f}  \n")
                    if 'num_shots' in item:
                        f.write(f"- Num Shots: {item['num_shots']}  \n")
                    if 'scale' in item:
                        f.write(f"- Scale: {item['scale']}  \n")
                    if 'scale_label' in item:
                        f.write(f"- Scale Label: {item['scale_label']}  \n")
                    f.write(f"\n<details>\n<summary>Description</summary>\n\n")
                    f.write(f"```\n{item.get('description', 'N/A')}\n```\n")
                    f.write(f"</details>\n\n")
                
                f.write(f"---\n\n")
    
    def _format_example_as_markdown(self, example: Dict, index: int) -> str:
        """將 example 格式化為 Markdown"""
        md = f"""# Example {index}: Rally {example['rally']}, Shot {example['shot']}

## Metadata

| Field | Value |
|-------|-------|
| Algorithm | {example['algorithm']} |
| Player | {example['player']} |
| Match ID | {example['match_id']} |
| Set | {example['set']} |
| Rally | {example['rally']} |
| Shot | {example['shot']} |
| Ball Round | {example['ball_round']} |
| Is Launch | {example['is_launch']} |
| Score | {example['metadata']['player_score']}-{example['metadata']['opponent_score']} |
| Player Location | {example['metadata']['player_location_area']} |
| Hit Position | {example['metadata']['hit_position_area']} |
| Opponent Shot | {example['metadata']['opponent_type_name']} |

## System Prompt

```markdown
{example['system_prompt']}
```

## User Prompt

```markdown
{example['user_prompt']}
```

## Retrieved Items ({len(example['retrieved_items'])})

"""
        
        for item in example['retrieved_items']:
            md += f"""### Item {item.get('index', '?')}

"""
            if 'distance' in item:
                md += f"- **Distance:** {item['distance']:.4f}\n"
            if 'distance' in item:
                md += f"- **Distance:** {item['distance']:.4f}\n"
            if 'similarity' in item:
                md += f"- **Similarity:** {item['similarity']:.4f}\n"
            if 'num_shots' in item:
                md += f"- **Num Shots:** {item['num_shots']}\n"
            if 'scale' in item:
                md += f"- **Scale:** {item['scale']}\n"
            if 'scale_label' in item:
                md += f"- **Scale Label:** {item['scale_label']}\n"
            
            md += f"\n**Description:**\n\n```\n{item.get('description', 'N/A')}\n```\n\n"
        
        md += """## Context Dict (Keys)

```json
"""
        # 只顯示 key 和類型，不顯示完整內容（避免過長）
        context_summary = {}
        for key, value in example['context_dict'].items():
            if isinstance(value, list):
                context_summary[key] = f"list (length={len(value)})"
            elif isinstance(value, str):
                context_summary[key] = f"str (length={len(value)})"
            elif isinstance(value, (int, float)):
                context_summary[key] = f"{type(value).__name__} = {value}"
            elif isinstance(value, bool):
                context_summary[key] = f"bool = {value}"
            else:
                context_summary[key] = f"{type(value).__name__}"
        
        md += json.dumps(context_summary, indent=2, default=str)
        md += """
```

## State Dict

```json
"""
        md += json.dumps(example['state_dict'], indent=2, ensure_ascii=False, default=str)
        md += """
```
"""
        
        return md


class NaiveRAGTracer(PromptTracer):
    """NaiveRAG Prompt Tracer"""
    
    def get_algorithm_name(self) -> str:
        return "naive_rag"
    
    def create_agent_components(self):
        """創建 NaiveRAG 組件"""
        from Agent.naive_rag import NaiveRAGDatabase, create_naive_rag_verbalizer
        from jinja2 import Template
        
        # 載入 train split
        train_split_path = self.config['paths'][f'{self.experiment}_split']
        with open(train_split_path, 'r') as f:
            train_split = json.load(f)
        
        train_matches = train_split.get('train', {}).get(self.player_name, [])
        
        if not train_matches:
            raise ValueError(f"No training matches found for player '{self.player_name}'")
        
        train_df = self.df[self.df['match_id'].isin(train_matches)].copy()
        
        # 創建 database
        self.database = NaiveRAGDatabase(train_df, self.player_name, self.player_id)
        
        # 創建 verbalizer
        self.verbalizer = create_naive_rag_verbalizer()
        
        # This tracer follows its existing classic retrieval flow.
        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "naive_rag")
        template_name = "shift.jinja" if self.experiment == "shifted" else "norm.jinja"
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            self.system_prompt = f.read()
        
        with open(os.path.join(prompt_dir, template_name), "r") as f:
            self.user_template = Template(f.read())
        
        # 初始化歷史
        self.state_history = []
    
    def retrieve_and_build_context(self, state, info, launch: bool) -> Dict[str, Any]:
        """執行 NaiveRAG 檢索並構建 context"""
        import numpy as np
        
        # 構建 query (連續特徵 + 離散特徵)
        query_continuous = [
            state[1][0], state[1][1],  # player_location_x, player_location_y
            state[3][0], state[3][1],  # hit_x, hit_y
            state[2][0], state[2][1],  # opponent_location_x, opponent_location_y
        ]
        query_discrete = [float(state[0])]  # opponent_type
        
        # 合併為單一 array [continuous..., discrete...]
        query_state = np.array(query_continuous + query_discrete, dtype=np.float32)
        
        # 檢索
        experiences = self.database.retrieve(query_state, top_k=self.top_k)
        
        # 構建 state_dict
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        
        state_dict = {
            'player': self.player_name,
            'is_target_turn': True,
            'turn_role': 'target',
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1],
        }
        
        # 構建 context
        normalized_history = []
        for stroke in self.state_history:
            stroke_data = dict(stroke)
            stroke_data['is_target_turn'] = str(stroke_data.get('player')) == str(self.player_name)
            stroke_data['turn_role'] = 'target' if stroke_data['is_target_turn'] else 'opponent'
            normalized_history.append(stroke_data)
        context = self.verbalizer.build_prompt_context_nl(
            state_dict, experiences, launch=launch, state_history=normalized_history
        )
        
        # 格式化 retrieved items
        retrieved_items = []
        for i, exp in enumerate(experiences, 1):
            retrieved_items.append({
                'index': i,
                'context': exp.get('context', {}),
                'action': exp.get('action', {}),
            })
        
        return {
            'retrieved_items': retrieved_items,
            'context_dict': context,
            'state_dict': state_dict
        }
    
    def reset_history(self):
        """重置歷史"""
        self.state_history = []
    
    def update_history(self, state_dict, info, launch):
        """更新歷史"""
        self.state_history.append(state_dict)

    def observe_history_row(self, row):
        info = self._build_info(row)
        self.state_history.append(self._build_state_dict(row, info))


class TimeRAGTracer(PromptTracer):
    """TimeRAG Prompt Tracer"""
    
    def __init__(self, player_name: str, experiment: str = 'norm', 
                 top_k: int = 5, num_examples: int = 5,
                 output_dir: str = './prompt_traces', window_size: int = 3):
        self.window_size = window_size
        super().__init__(player_name, experiment, top_k, num_examples, output_dir)
    
    def get_algorithm_name(self) -> str:
        return "timerag"
    
    def create_agent_components(self):
        """創建 TimeRAG 組件"""
        from Agent.timerag import TimeRAGDatabase, create_timerag_verbalizer
        from jinja2 import Template
        
        # 載入 train split
        train_split_path = self.config['paths'][f'{self.experiment}_split']
        with open(train_split_path, 'r') as f:
            train_split = json.load(f)
        
        train_matches = train_split.get('train', {}).get(self.player_name, [])
        
        if not train_matches:
            raise ValueError(f"No training matches found for player '{self.player_name}'")
        
        train_df = self.df[self.df['match_id'].isin(train_matches)].copy()
        
        # 創建 database
        self.database = TimeRAGDatabase(
            train_df, self.player_name, self.player_id, 
            window_size=self.window_size
        )
        
        # 創建 verbalizer
        self.verbalizer = create_timerag_verbalizer()
        
        # This tracer follows its existing classic retrieval flow.
        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "timerag")
        template_name = "shift.jinja" if self.experiment == "shifted" else "norm.jinja"
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            self.system_prompt = f.read()
        
        with open(os.path.join(prompt_dir, template_name), "r") as f:
            self.user_template = Template(f.read())
        
        # 初始化歷史
        self.state_history = []
    
    def retrieve_and_build_context(self, state, info, launch: bool) -> Dict[str, Any]:
        """執行 TimeRAG 檢索並構建 context"""
        # 構建 query window
        current_continuous = [
            state[1][0], state[1][1],  # player_location_x, player_location_y
            state[3][0], state[3][1],  # hit_x, hit_y
            state[2][0], state[2][1],  # opponent_location_x, opponent_location_y
        ]
        current_discrete = [float(state[0])]  # opponent_type
        
        # 添加到歷史
        self.state_history.append({
            'continuous': current_continuous,
            'discrete': current_discrete
        })
        
        # 保持 window_size 個最新狀態
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
        
        # 構建 query window（前面補零）
        import numpy as np
        query_continuous = np.zeros((self.window_size, len(current_continuous)), dtype=np.float32)
        query_discrete = np.zeros((self.window_size, len(current_discrete)), dtype=np.float32)
        
        n_history = len(self.state_history)
        for i in range(n_history):
            query_continuous[self.window_size - n_history + i] = self.state_history[i]['continuous']
            query_discrete[self.window_size - n_history + i] = self.state_history[i]['discrete']
        
        # 檢索
        windows = self.database.retrieve(query_continuous, query_discrete, top_k=self.top_k)
        
        # 構建 state_dict
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        
        state_dict = {
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1],
        }
        
        # 構建 context
        context = self.verbalizer.build_prompt_context_nl(
            state_dict, windows, launch=launch, 
            window_size=self.window_size, state_history=self.state_history[:-1]
        )
        
        # 格式化 retrieved items
        retrieved_items = []
        for i, win in enumerate(windows, 1):
            formatted = self.verbalizer.format_stage2_case(win, i)
            retrieved_items.append({
                'index': formatted['index'],
                'num_shots': formatted['num_shots'],
                'stage2_description': formatted['stage2_description'],
            })
        
        return {
            'retrieved_items': retrieved_items,
            'context_dict': context,
            'state_dict': state_dict
        }
    
    def reset_history(self):
        """重置歷史"""
        self.state_history = []
    
    def update_history(self, state_dict, info, launch):
        """更新歷史（已在 retrieve_and_build_context 中處理）"""
        pass


class TwinTracer(PromptTracer):
    """Twin (I-MSDP) Prompt Tracer"""
    
    def __init__(self, player_name: str, experiment: str = 'norm', 
                 top_k: int = 5, num_examples: int = 5,
                 output_dir: str = './prompt_traces', max_scale: int = 10):
        self.max_scale = max_scale
        self.prev_distance_tensor = None
        super().__init__(player_name, experiment, top_k, num_examples, output_dir)
    
    def get_algorithm_name(self) -> str:
        return "twin"
    
    def create_agent_components(self):
        """創建 Twin 組件"""
        from Agent.twin import TwinDatabase, create_twin_verbalizer
        from jinja2 import Template
        
        # 載入 train split
        train_split_path = self.config['paths'][f'{self.experiment}_split']
        with open(train_split_path, 'r') as f:
            train_split = json.load(f)
        
        train_matches = train_split.get('train', {}).get(self.player_name, [])
        
        if not train_matches:
            raise ValueError(f"No training matches found for player '{self.player_name}'")
        
        train_df = self.df[self.df['match_id'].isin(train_matches)].copy()
        
        # 創建 database
        self.database = TwinDatabase(
            train_df, self.player_name, self.player_id,
            max_scale=self.max_scale
        )
        
        # 創建 verbalizer
        self.verbalizer = create_twin_verbalizer()
        
        # This tracer follows its existing classic retrieval flow.
        prompt_root = os.path.join(project_path, "Prompts", "New")
        prompt_dir = os.path.join(prompt_root, "twin")
        template_name = "classic_shifted_user.jinja" if self.experiment == "shifted" else "classic_user.jinja"
        with open(os.path.join(prompt_root, "simulate_system.md"), "r") as f:
            self.system_prompt = f.read()
        
        with open(os.path.join(prompt_dir, template_name), "r") as f:
            self.user_template = Template(f.read())
        
        # 初始化
        self.state_history = []
        self._current_rally = None
    
    def retrieve_and_build_context(self, state, info, launch: bool) -> Dict[str, Any]:
        """執行 Twin 檢索並構建 context"""
        import numpy as np
        
        # 檢查 rally 是否改變
        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.prev_distance_tensor = None
            self.state_history = []
        
        # 構建 query state (必須是 dict，包含正確的 keys)
        query_state = {
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1],
            'opponent_type': int(state[0])
        }
        
        # 增量檢索
        segments, self.prev_distance_tensor = self.database.incremental_retrieve(
            query_state, self.prev_distance_tensor, top_k=self.top_k
        )
        
        # 構建 state_dict
        player_score = info['score'][0] if info.get('score') else 0
        opponent_score = info['score'][1] if info.get('score') else 0
        
        state_dict = {
            'rally': info.get('rally', 0),
            'ball_round': info['round'][-1] if info.get('round') else 1,
            'player_score': player_score,
            'opponent_score': opponent_score,
            'score_status': int(player_score - opponent_score),
            'player_location_x': state[1][0],
            'player_location_y': state[1][1],
            'hit_x': state[3][0],
            'hit_y': state[3][1],
            'opponent_type': int(state[0]),
            'opponent_location_x': state[2][0],
            'opponent_location_y': state[2][1],
        }
        
        # 構建 context
        context = self.verbalizer.build_prompt_context_nl(
            state_dict, segments, launch=launch, state_history=self.state_history
        )
        
        # 格式化 retrieved items
        retrieved_items = []
        for i, seg in enumerate(segments, 1):
            formatted = self.verbalizer.format_segment_turn_based(seg, i)
            retrieved_items.append({
                'index': formatted['index'],
                'scale': formatted['scale'],
                'scale_label': formatted['scale_label'],
                'distance': formatted['distance'],
                'num_shots': formatted['num_shots'],
                'description': formatted['description'],
            })
        
        return {
            'retrieved_items': retrieved_items,
            'context_dict': context,
            'state_dict': state_dict
        }
    
    def reset_history(self):
        """重置歷史"""
        self.state_history = []
        self._current_rally = None
        self.prev_distance_tensor = None
    
    def update_history(self, state_dict, info, launch):
        """更新歷史"""
        rally_num = info.get('rally', 0)
        if self._current_rally != rally_num:
            self._current_rally = rally_num
            self.state_history = []
        self.state_history.append(state_dict)


def create_tracer(algorithm: str, **kwargs) -> PromptTracer:
    """工廠函數：創建對應的 Tracer"""
    algorithm = algorithm.lower()
    
    if algorithm == 'naive_rag' or algorithm == 'naiverag':
        return NaiveRAGTracer(**kwargs)
    elif algorithm == 'timerag':
        return TimeRAGTracer(**kwargs)
    elif algorithm == 'twin':
        return TwinTracer(**kwargs)
    else:
        raise ValueError(f"不支援的算法: {algorithm}. 支援: naive_rag, timerag, twin")


def main():
    parser = argparse.ArgumentParser(
        description='Trace RAG Prompts - 預覽 NaiveRAG、TimeRAG、Twin 的 Prompt 結構',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # Trace TimeRAG
  python Scripts/single_player_eval/trace_prompts.py \\
      --algorithm timerag \\
      --player "Kento MOMOTA" \\
      --window-size 3 \\
      --num-examples 5

  # Trace Twin
  python Scripts/single_player_eval/trace_prompts.py \\
      --algorithm twin \\
      --player "Kento MOMOTA" \\
      --max-scale 10 \\
      --num-examples 5

  # Trace NaiveRAG
  python Scripts/single_player_eval/trace_prompts.py \\
      --algorithm naive_rag \\
      --player "Viktor AXELSEN" \\
      --top-k 5 \\
      --num-examples 3
        """
    )
    
    # 主要參數
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['naive_rag', 'timerag', 'twin'],
        help='要追蹤的算法'
    )
    
    parser.add_argument(
        '--player',
        type=str,
        required=True,
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
        '--top-k',
        type=int,
        default=None,
        help='檢索數量（預設: config retrieval.top_k）'
    )
    
    parser.add_argument(
        '--num-examples',
        type=int,
        default=5,
        help='要生成的 example 數量（預設: 5）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./prompt_traces',
        help='輸出目錄（預設: ./prompt_traces）'
    )
    
    # 算法特定參數
    parser.add_argument(
        '--window-size',
        type=int,
        default=3,
        help='TimeRAG 時間窗口大小（預設: 3）'
    )
    
    parser.add_argument(
        '--max-scale',
        type=int,
        default=10,
        help='Twin I-MSDP 最大 scale（預設: 10）'
    )
    
    args = parser.parse_args()

    if args.algorithm in {'naive_rag', 'timerag', 'twin'}:
        parser.error(
            'NaiveRAG/TimeRAG/Twin now use stage-specific prompts. Run '
            'eval_all_rallies.py with --enable-tracing to inspect both retrieval '
            'and generation stages in Langfuse.'
        )

    import yaml
    with open(os.path.join(project_path, 'Scripts', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    if args.top_k is None:
        args.top_k = config.get('retrieval', {}).get('top_k', 5)

    print_runtime_box(
        "Runtime Parameters: trace_prompts",
        sorted(vars(args).items()),
    )
    
    # 收集參數
    kwargs = {
        'player_name': args.player,
        'experiment': args.experiment,
        'top_k': args.top_k,
        'num_examples': args.num_examples,
        'output_dir': args.output_dir,
    }
    
    # 算法特定參數
    if args.algorithm == 'timerag':
        kwargs['window_size'] = args.window_size
    elif args.algorithm == 'twin':
        kwargs['max_scale'] = args.max_scale
    
    try:
        # 創建 tracer
        tracer = create_tracer(args.algorithm, **kwargs)
        
        # 執行追蹤
        examples = tracer.trace_prompts()
        
        print(f"✅ 成功生成 {len(examples)} 個 prompt examples!")
        return 0
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
