from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Utils.helper import action_col, state_col
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key
from Utils.artifact_names import player_database_filename


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_prepared_directory(staging_dir, output_dir):
    """Replace a live database directory only after a complete successful build."""
    output_dir = os.path.abspath(output_dir)
    backup_dir = None
    if os.path.exists(output_dir):
        backup_dir = tempfile.mkdtemp(
            prefix=f'.{os.path.basename(output_dir)}_backup_',
            dir=os.path.dirname(output_dir),
        )
        os.rmdir(backup_dir)
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if backup_dir and os.path.exists(backup_dir) and not os.path.exists(output_dir):
            os.replace(backup_dir, output_dir)
        raise
    finally:
        if backup_dir and os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)


def load_preparation_data(merged_csv, split_path, player_map_path, profiles_path=None,
                          drop_incomplete_rows=True):
    """Load and filter the shared inputs used by the baseline data builders."""
    split_content = load_rally_dataset(split_path, profiles_path)
    with open(player_map_path, 'r') as f:
        player2index = json.load(f)

    df = pd.read_csv(merged_csv)
    required_cols = state_col + action_col + ['match_id', 'server']
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {merged_csv}: {missing_cols}")

    base_cols = list(required_cols)
    for optional_col in ('set', 'moving_distance', 'getpoint_player'):
        if optional_col in df.columns:
            base_cols.append(optional_col)
    df = df[base_cols]
    if drop_incomplete_rows:
        df = df.dropna(subset=state_col + action_col)
    df = df.reset_index(drop=True)
    return df, split_content, player2index


def load_match_preparation_data(merged_csv, split_path, player_map_path):
    """Load match-level inputs still used by non-BC/ShuttleNet methods."""
    with open(split_path, 'r') as f:
        split_content = json.load(f)
    with open(player_map_path, 'r') as f:
        player2index = json.load(f)

    df = pd.read_csv(merged_csv)
    required_cols = state_col + action_col + ['match_id', 'server']
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {merged_csv}: {missing_cols}")

    base_cols = list(required_cols)
    for optional_col in ('set', 'moving_distance', 'getpoint_player'):
        if optional_col in df.columns:
            base_cols.append(optional_col)
    df = df[base_cols].dropna(subset=state_col + action_col).reset_index(drop=True)
    return df, split_content, player2index


def prepare_bc_data(df, split_content, player2index, output_dir):
    """
    從 train split 提取 BC 訓練資料。
    
    Args:
        df: DataFrame with merged data (player column should be mapped to IDs)
        split_content: Train/test split dict
        player2index: Player name to ID mapping
        output_dir: Directory to save output files
    
    Output:
        - train_state.pkl: numpy array of states
        - train_action.pkl: numpy array of actions
    """
    state_data, action_data = [], []
    
    # Ensure player column contains IDs (integers), not names
    if df['player'].dtype == object:
        print("Converting player names to IDs for BC data preparation...")
        df = df.copy()
        # Assume it's already mapped or map it now
        if df['player'].iloc[0] in player2index:
            # Contains names, map to IDs
            df['player'] = df['player'].map(player2index)
    
    selected_df = filter_rally_rows(df, split_content["train"])
    if selected_df.empty:
        raise ValueError("No BC rows matched the rally-level training split")
    if selected_df['player'].isna().any():
        raise ValueError("BC training rows contain players missing from player_index_map")

    state_data.extend(selected_df[state_col].values)
    action_data.extend(selected_df[action_col].values)

    expected_keys = {rally_key(record) for record in split_content['train']}
    matched_keys = set(
        selected_df[['match_id', 'set', 'rally']]
        .astype(int)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    skipped_keys = sorted(expected_keys - matched_keys)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle
    state_path = os.path.join(output_dir, "train_state.pkl")
    action_path = os.path.join(output_dir, "train_action.pkl")
    
    with open(state_path, 'wb') as f:
        pickle.dump(np.array(state_data), f)
    with open(action_path, 'wb') as f:
        pickle.dump(np.array(action_data), f)
    with open(os.path.join(output_dir, 'preparation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'physical_train_rallies_in_split': len(expected_keys),
            'physical_train_rallies_matched': len(matched_keys),
            'physical_train_rallies_skipped': len(skipped_keys),
            'skipped_rally_keys': [
                {'match_id': key[0], 'set': key[1], 'rally': key[2]}
                for key in skipped_keys
            ],
            'training_samples': len(state_data),
        }, f, ensure_ascii=False, indent=2)
    
    print(f"BC data saved to {output_dir}")
    print(f"  - train_state.pkl: {len(state_data)} samples")
    print(f"  - train_action.pkl: {len(action_data)} samples")
    print(f"  - physical rallies: {len(matched_keys)}/{len(expected_keys)} matched")


def prepare_naive_rag_data(df, split_content, player2index, output_dir):
    """
    為每個 train 中的玩家構建 NaiveRAGDatabase。
    
    Args:
        df: DataFrame with merged data (player column should be player names)
        split_content: Train/test split dict
        player2index: Player name to ID mapping
        output_dir: Directory to save output files
    
    Output:
        - {player_name}_db.pkl: One database file per player
        - preparation_manifest.json: rally split provenance and build counts
    """
    # Import here to avoid circular import issues
    from Agent.naive_rag import NaiveRAGDatabase
    retrieval_features = [
        'player_location_x', 'player_location_y',
        'hit_x', 'hit_y',
        'opponent_location_x', 'opponent_location_y',
        'ball_distance_x', 'ball_distance_y',
    ]
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    staging_dir = tempfile.mkdtemp(
        prefix=f'.{os.path.basename(output_dir)}_build_',
        dir=os.path.dirname(output_dir),
    )
    
    # Define all columns needed by NaiveRAGDatabase
    # These are based on state_col and action_col from Utils.helper
    required_cols = [
        'player',  # Must be player names, not IDs
        'match_id', 'set',
        # state_col columns
        'rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
        'player_location_x', 'player_location_y',
        'ball_distance', 'ball_distance_x', 'ball_distance_y', 'hit_x', 'hit_y',
        'opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y',
        # action_col columns
        'landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y'
    ]
    # Check if all required columns are present
    available_cols = [c for c in required_cols if c in df.columns]
    missing_cols = set(required_cols) - set(available_cols)
    if missing_cols:
        raise ValueError(f"Missing NaiveRAG columns: {sorted(missing_cols)}")
    
    # Pandas may use StringDtype rather than object for player names.
    player_values = df['player'].dropna()
    first_player = str(player_values.iloc[0]).strip() if len(player_values) else ""
    df = df.copy()
    if first_player in player2index:
        df['player'] = df['player'].astype(str).str.strip()
    else:
        print("Warning: player column appears to contain IDs, mapping back to names...")
        index2player = {int(v): k for k, v in player2index.items()}

        def map_player_id(value):
            try:
                return index2player.get(int(float(value)))
            except (TypeError, ValueError):
                return None

        df['player'] = df['player'].map(map_player_id)
    
    # Filter to only include required columns that exist
    df_filtered = df[available_cols].copy()
    
    train_records = split_content.get('train', [])
    test_records = split_content.get('test', [])
    required_players = sorted({
        player
        for record in train_records + test_records
        for player in record.get('players', [])
    })
    successful_builds = 0
    failed_players = []
    player_summaries = {}
    
    for player_name in tqdm(required_players, desc="Building NaiveRAG databases"):
        if player_name not in player2index:
            print(f"Warning: {player_name} not in player_index_map, skipping")
            failed_players.append(player_name)
            continue
        
        player_id = player2index[player_name]
        player_records = [
            record for record in train_records if player_name in record.get('players', [])
        ]
        player_match_df = filter_rally_rows(df_filtered, player_records)
        player_data = player_match_df[player_match_df['player'] == player_name].copy()
        
        if len(player_data) == 0:
            print(f"Warning: No data found for {player_name}, skipping")
            failed_players.append(player_name)
            continue
        
        try:
            build_metadata = {
                'dataset': 'rally',
                'split_sha256': split_content.get('split_sha256'),
                'profiles_sha256': split_content.get('profiles_sha256'),
                'train_rally_keys': [list(rally_key(record)) for record in player_records],
                'train_rally_count': len(player_records),
                'training_sample_count': len(player_data),
                'retrieval_features': retrieval_features,
            }
            database = NaiveRAGDatabase(
                player_match_df,
                player_name,
                player_id,
                build_metadata=build_metadata,
            )
            
            # Save database
            output_path = os.path.join(staging_dir, player_database_filename(player_name))
            database.save(output_path)
            successful_builds += 1
            player_summaries[player_name] = {
                'database_file': os.path.basename(output_path),
                'database_sha256': _file_sha256(output_path),
                'train_rally_count': len(player_records),
                'training_sample_count': len(player_data),
            }
        except Exception as e:
            print(f"Error building database for {player_name}: {e}")
            failed_players.append(player_name)
            continue
    
    with open(os.path.join(staging_dir, 'preparation_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'schema_version': 3,
            'algorithm': 'naive_rag',
            'database_schema_version': NaiveRAGDatabase.schema_version,
            'verbalization_schema_version': NaiveRAGDatabase.verbalization_schema_version,
            'retrieval_architecture': 'two_stage_v1',
            'retrieval_features': retrieval_features,
            'dataset': 'rally',
            'split_sha256': split_content.get('split_sha256'),
            'profiles_sha256': split_content.get('profiles_sha256'),
            'physical_train_rallies': len(train_records),
            'physical_test_rallies': len(test_records),
            'players': player_summaries,
        }, f, ensure_ascii=False, indent=2)
        f.write('\n')

    if failed_players:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to build NaiveRAG databases for: {failed_players}")
    _publish_prepared_directory(staging_dir, output_dir)
    print(f"\nSuccessfully built {successful_builds} NaiveRAG databases")
    print(f"All databases saved to: {output_dir}")
    return successful_builds


def prepare_timerag_data(df, split_content, player2index, output_dir, window_size=3):
    """
    為每個 train 中的玩家構建 TimeRAGDatabase。
    
    Args:
        df: DataFrame with merged data (player column should be player names)
        split_content: Train/test split dict
        player2index: Player name to ID mapping
        output_dir: Directory to save output files
        window_size: Size of time window for retrieval (default: 3)
    
    Output:
        - {player_name}_db.pkl: One database file per player
    """
    # Import here to avoid circular import issues
    from Agent.timerag import TimeRAGDatabase
    retrieval_features = [
        'player_location_x', 'player_location_y',
        'hit_x', 'hit_y',
        'opponent_location_x', 'opponent_location_y',
        'ball_distance_x', 'ball_distance_y',
    ]
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    staging_dir = tempfile.mkdtemp(
        prefix=f'.{os.path.basename(output_dir)}_build_',
        dir=os.path.dirname(output_dir),
    )
    
    # Define all columns needed by TimeRAGDatabase
    # Note: 'set' column is optional and may not be present in all datasets
    required_cols = [
        'player',  # Must be player names, not IDs
        'match_id', 'set',
        # state_col columns
        'rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
        'player_location_x', 'player_location_y',
        'ball_distance', 'ball_distance_x', 'ball_distance_y', 'hit_x', 'hit_y',
        'opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y',
        # action_col columns
        'landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y'
    ]
    optional_cols = []
    
    # Check if all required columns are present
    available_cols = [c for c in required_cols + optional_cols if c in df.columns]
    missing_cols = set(required_cols) - set(available_cols)
    if missing_cols:
        raise ValueError(f"Missing TimeRAG columns: {sorted(missing_cols)}")
    
    # Pandas may use StringDtype rather than object for player names.
    player_values = df['player'].dropna()
    first_player = str(player_values.iloc[0]).strip() if len(player_values) else ""
    df = df.copy()
    if first_player in player2index:
        df['player'] = df['player'].astype(str).str.strip()
    else:
        print("Warning: player column appears to contain IDs, mapping back to names...")
        index2player = {int(v): k for k, v in player2index.items()}

        def map_player_id(value):
            try:
                return index2player.get(int(float(value)))
            except (TypeError, ValueError):
                return None

        df['player'] = df['player'].map(map_player_id)
    
    # Filter to only include required columns that exist
    df_filtered = df[available_cols].copy()
    
    train_records = split_content.get('train', [])
    test_records = split_content.get('test', [])
    required_players = sorted({
        player
        for record in train_records + test_records
        for player in record.get('players', [])
    })
    successful_builds = 0
    failed_players = []
    player_summaries = {}
    
    for player_name in tqdm(required_players, desc=f"Building TimeRAG databases (window_size={window_size})"):
        if player_name not in player2index:
            print(f"Warning: {player_name} not in player_index_map, skipping")
            failed_players.append(player_name)
            continue
        
        player_id = player2index[player_name]
        player_records = [
            record for record in train_records if player_name in record.get('players', [])
        ]
        player_match_df = filter_rally_rows(df_filtered, player_records)
        player_data = player_match_df[player_match_df['player'] == player_name].copy()
        
        if len(player_data) == 0:
            print(f"Warning: No data found for {player_name}, skipping")
            failed_players.append(player_name)
            continue
        
        try:
            build_metadata = {
                'dataset': 'rally',
                'split_sha256': split_content.get('split_sha256'),
                'profiles_sha256': split_content.get('profiles_sha256'),
                'train_rally_keys': [list(rally_key(record)) for record in player_records],
                'train_rally_count': len(player_records),
                'training_sample_count': len(player_data),
                'window_size': window_size,
                'retrieval_features': retrieval_features,
            }
            database = TimeRAGDatabase(
                player_match_df,
                player_name,
                player_id,
                window_size=window_size,
                build_metadata=build_metadata,
            )
            
            # Save database
            output_path = os.path.join(staging_dir, player_database_filename(player_name))
            database.save(output_path)
            successful_builds += 1
            player_summaries[player_name] = {
                'database_file': os.path.basename(output_path),
                'database_sha256': _file_sha256(output_path),
                'train_rally_count': len(player_records),
                'window_count': len(database.windows),
            }
        except Exception as e:
            print(f"Error building database for {player_name}: {e}")
            traceback.print_exc()
            failed_players.append(player_name)
            continue
    
    with open(os.path.join(staging_dir, 'preparation_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'schema_version': 3,
            'algorithm': 'timerag',
            'database_schema_version': TimeRAGDatabase.schema_version,
            'verbalization_schema_version': TimeRAGDatabase.verbalization_schema_version,
            'retrieval_architecture': 'two_stage_v1',
            'retrieval_features': retrieval_features,
            'dataset': 'rally',
            'split_sha256': split_content.get('split_sha256'),
            'profiles_sha256': split_content.get('profiles_sha256'),
            'window_size': window_size,
            'physical_train_rallies': len(train_records),
            'physical_test_rallies': len(test_records),
            'players': player_summaries,
        }, f, ensure_ascii=False, indent=2)
        f.write('\n')

    if failed_players:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to build TimeRAG databases for: {failed_players}")
    _publish_prepared_directory(staging_dir, output_dir)
    print(f"\nSuccessfully built {successful_builds} TimeRAG databases")
    print(f"All databases saved to: {output_dir}")
    return successful_builds


def prepare_twin_data(
    df,
    split_content,
    player2index,
    output_dir,
    use_llm=True,
    max_rallies_per_player=None,
    twin_workers=1,
    llm_config_path=None,
):
    """
    為每個 train 中的玩家構建 Twin aspect-memory database。

    Args:
        df: DataFrame with merged data (player column should be player names)
        split_content: Train/test split dict
        player2index: Player name to ID mapping
        output_dir: Directory to save output files
        use_llm: Whether to annotate rallies with the LLM or use fallback phases
        max_rallies_per_player: Optional smoke-test cap; None means all train rallies
        twin_workers: Number of players to build in parallel
        llm_config_path: Config file used to initialize each worker's LLM client

    Output:
        - {player_name}_db.pkl: One database file per player
    """
    # Import here to avoid circular import issues
    from Agent.twin import TwinDatabase

    output_dir = os.path.abspath(output_dir)
    output_parent = os.path.dirname(output_dir)
    os.makedirs(output_parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix='.twin_memory_', dir=output_parent)

    # Define all columns needed by TwinDatabase
    required_cols = [
        'player',  # Must be player names, not IDs
        'match_id',
        # state_col columns
        'rally', 'ball_round', 'player_score', 'opponent_score', 'score_status',
        'player_location_x', 'player_location_y',
        'ball_distance', 'ball_distance_x', 'ball_distance_y', 'hit_x', 'hit_y',
        'opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y',
        # action_col columns
        'landing_x', 'landing_y', 'player_type', 'moving_x', 'moving_y'
    ]
    if 'set' in df.columns:
        required_cols.insert(2, 'set')
    if 'moving_distance' in df.columns:
        required_cols.append('moving_distance')
    if 'getpoint_player' in df.columns:
        required_cols.append('getpoint_player')

    # Check if all required columns are present
    available_cols = [c for c in required_cols if c in df.columns]
    missing_cols = set(required_cols) - set(available_cols)
    if missing_cols:
        raise ValueError(f"Missing Twin columns: {sorted(missing_cols)}")

    # Ensure player column contains names (strings), not IDs. Pandas may load
    # string columns as ArrowStringDtype, so do not rely on dtype == object.
    player_values = df['player'].dropna()
    first_player = str(player_values.iloc[0]).strip() if len(player_values) else ""
    df = df.copy()
    if first_player in player2index:
        df['player'] = df['player'].astype(str).str.strip()
    else:
        print("Warning: player column appears to contain IDs, mapping back to names...")
        index2player = {int(v): k for k, v in player2index.items()}

        def map_player_id(value):
            try:
                return index2player.get(int(float(value)))
            except (TypeError, ValueError):
                return None

        df['player'] = df['player'].map(map_player_id)

    # Filter to only include required columns that exist
    df_filtered = df[available_cols].copy()

    train_records = list(split_content.get('train', []))
    test_records = list(split_content.get('test', []))
    required_players = sorted({
        player
        for record in train_records + test_records
        for player in record.get('players', [])
    })
    successful_builds = 0
    failed_players = []
    player_summaries = {}
    twin_workers = max(1, int(twin_workers or 1))

    def build_player_database(player_name):
        player_name = str(player_name).strip()
        if player_name not in player2index:
            return False, None, f"Warning: {player_name} not in player_index_map, skipping"

        player_id = player2index[player_name]
        player_records = [
            record for record in train_records if player_name in record.get('players', [])
        ]

        # Include both players' rows from the target player's train rallies so
        # rally-level LLM annotations see the full exchange. Only target-player
        # strokes are converted into memories inside TwinDatabase.
        player_match_df = filter_rally_rows(df_filtered, player_records)
        incomplete_keys = set(
            player_match_df.loc[
                player_match_df[state_col + action_col].isna().any(axis=1),
                ['match_id', 'set', 'rally'],
            ]
            .astype(int)
            .itertuples(index=False, name=None)
        )
        if incomplete_keys:
            player_records = [
                record for record in player_records
                if rally_key(record) not in incomplete_keys
            ]
            player_match_df = filter_rally_rows(df_filtered, player_records)
        player_data = player_match_df[player_match_df['player'] == player_name].copy()

        if len(player_data) == 0:
            return False, None, f"Warning: No rally-level train data found for {player_name}, skipping"

        try:
            database = TwinDatabase(
                player_match_df,
                player_name,
                player_id,
            )
            if use_llm:
                from Utils.llm import LLM

                database.build_with_llm(
                    llm=LLM(config_path=llm_config_path),
                    max_rallies=max_rallies_per_player,
                )
            else:
                database.build_with_fallback(max_rallies=max_rallies_per_player)

            database.build_metadata = {
                'dataset': 'rally',
                'split_sha256': split_content.get('split_sha256'),
                'profiles_sha256': split_content.get('profiles_sha256'),
                'use_llm': bool(use_llm),
                'max_rallies_per_player': max_rallies_per_player,
                'train_rally_keys': [list(rally_key(record)) for record in player_records],
                'train_rally_count': len(player_records),
                'skipped_incomplete_rally_keys': [list(key) for key in sorted(incomplete_keys)],
                'target_source_row_count': int(len(player_data)),
                'memory_count': int(len(database.memories)),
            }

            output_path = os.path.join(staging_dir, player_database_filename(player_name))
            database.save(output_path)
            summary = {
                'database_file': os.path.basename(output_path),
                'database_sha256': _file_sha256(output_path),
                'train_rally_count': len(player_records),
                'skipped_incomplete_rally_count': len(incomplete_keys),
                'training_sample_count': int(len(player_data)),
                'memory_count': int(len(database.memories)),
            }
            return True, summary, None
        except Exception as e:
            return False, None, f"Error building database for {player_name}: {e}\n{traceback.format_exc()}"

    if twin_workers == 1:
        for player_name in tqdm(required_players, desc="Building Twin aspect-memory databases"):
            ok, summary, message = build_player_database(player_name)
            if ok:
                successful_builds += 1
                player_summaries[player_name] = summary
            elif message:
                print(message)
                failed_players.append(player_name)
    else:
        with ThreadPoolExecutor(max_workers=twin_workers) as executor:
            futures = {
                executor.submit(build_player_database, player_name): player_name
                for player_name in required_players
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building Twin aspect-memory databases"):
                try:
                    ok, summary, message = future.result()
                except Exception as exc:
                    ok = False
                    summary = None
                    message = f"Error building database for {futures[future]}: {exc}\n{traceback.format_exc()}"
                if ok:
                    successful_builds += 1
                    player_summaries[futures[future]] = summary
                elif message:
                    print(message)
                    failed_players.append(futures[future])

    if failed_players:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to build Twin databases for: {failed_players}")
    if successful_builds == 0:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError('Twin memory preparation produced no player databases')

    memory_prompt = os.path.join(project_path, 'Prompts', 'New', 'twin', 'memory_user.jinja')
    system_prompt = os.path.join(project_path, 'Prompts', 'New', 'simulate_system.md')
    from Utils.llm import LLM

    llm_settings = LLM.resolve_effective_settings(config_path=llm_config_path)
    safe_llm_settings = {
        'provider': llm_settings.get('provider'),
        'llm_type': llm_settings.get('llm_type'),
        'model_name': llm_settings.get('model_name'),
        'generation_params': llm_settings.get('generation_params'),
        'deterministic_active': llm_settings.get('deterministic_active'),
    }
    with open(os.path.join(staging_dir, 'preparation_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'schema_version': 1,
            'algorithm': 'twin',
            'database_schema_version': TwinDatabase.schema_version,
            'retrieval_architecture': 'twin_aspect_two_stage_v1',
            'dataset': 'rally',
            'split_sha256': split_content.get('split_sha256'),
            'profiles_sha256': split_content.get('profiles_sha256'),
            'memory_prompt_sha256': _file_sha256(memory_prompt),
            'system_prompt_sha256': _file_sha256(system_prompt),
            'llm': safe_llm_settings if use_llm else None,
            'use_llm': bool(use_llm),
            'max_rallies_per_player': max_rallies_per_player,
            'physical_train_rallies': len(train_records),
            'physical_test_rallies': len(test_records),
            'players': player_summaries,
        }, f, ensure_ascii=False, indent=2)
        f.write('\n')

    _publish_prepared_directory(staging_dir, output_dir)

    print(f"\nSuccessfully built {successful_builds} Twin databases")
    print(f"All databases saved to: {output_dir}")
    return successful_builds


def prepare_twin_embeddings(
    memory_dir,
    cache_dir,
    embedding_model_name,
    embedding_device='auto',
    embedding_batch_size=None,
    embedding_revision=None,
    normalize_embeddings=True,
    embedding_dtype='float32',
    players=None,
    force=False,
    embedding_model=None,
    embedding_model_loader=None,
    embedding_provider='local',
    embedding_dimension=None,
):
    """Build model-specific embedding caches from canonical Twin memories."""
    from Agent.twin import (
        TwinDatabase,
        TwinEmbeddingCache,
        embedding_cache_paths,
        twin_memory_digest,
    )

    if not os.path.isdir(memory_dir):
        raise FileNotFoundError(f"Twin memory directory does not exist: {memory_dir}")
    database_files = sorted(
        filename for filename in os.listdir(memory_dir) if filename.endswith('_db.pkl')
    )
    if not database_files:
        raise FileNotFoundError(f"No canonical Twin memories found in: {memory_dir}")

    selected_players = {str(player).strip() for player in players or []}
    built = 0
    skipped = 0
    loaded_model = embedding_model
    player_summaries = {}

    for filename in tqdm(database_files, desc='Building Twin embedding caches'):
        database_path = os.path.join(memory_dir, filename)
        database = TwinDatabase.load(database_path)
        if selected_players and database.player_name not in selected_players:
            continue

        if not force:
            try:
                TwinEmbeddingCache.load(
                    cache_dir,
                    database,
                    embedding_model_name,
                    revision=embedding_revision,
                    normalize_embeddings=normalize_embeddings,
                    dtype=embedding_dtype,
                    mmap_mode='r',
                    provider=embedding_provider,
                    embedding_dimension=embedding_dimension,
                )
                array_path, metadata_path = embedding_cache_paths(
                    cache_dir,
                    database.player_name,
                    embedding_model_name,
                    embedding_revision,
                    normalize_embeddings,
                    embedding_dtype,
                    embedding_provider,
                    embedding_dimension,
                )
                player_summaries[database.player_name] = {
                    'database_file': filename,
                    'memory_digest': twin_memory_digest(database.memories),
                    'embedding_file': os.path.relpath(array_path, cache_dir),
                    'embedding_sha256': _file_sha256(array_path),
                    'metadata_file': os.path.relpath(metadata_path, cache_dir),
                    'metadata_sha256': _file_sha256(metadata_path),
                }
                skipped += 1
                continue
            except (FileNotFoundError, KeyError, OSError, ValueError):
                pass

        if loaded_model is None:
            if embedding_model_loader is not None:
                loaded_model = embedding_model_loader()
            else:
                raise ValueError(
                    'Twin embedding preparation requires embedding_model or '
                    'embedding_model_loader'
                )
        cache = TwinEmbeddingCache.build(
            database,
            loaded_model,
            embedding_model_name,
            revision=embedding_revision,
            batch_size=embedding_batch_size,
            normalize_embeddings=normalize_embeddings,
            dtype=embedding_dtype,
            provider=embedding_provider,
            embedding_dimension=embedding_dimension,
        )
        array_path, metadata_path = cache.save(cache_dir)
        player_summaries[database.player_name] = {
            'database_file': filename,
            'memory_digest': cache.memory_digest,
            'embedding_file': os.path.relpath(array_path, cache_dir),
            'embedding_sha256': _file_sha256(array_path),
            'metadata_file': os.path.relpath(metadata_path, cache_dir),
            'metadata_sha256': _file_sha256(metadata_path),
        }
        built += 1

    if selected_players and built + skipped == 0:
        raise ValueError(f"No Twin memories matched players: {sorted(selected_players)}")
    first_database = TwinDatabase.load(os.path.join(memory_dir, database_files[0]))
    first_array_path, _ = embedding_cache_paths(
        cache_dir,
        first_database.player_name,
        embedding_model_name,
        embedding_revision,
        normalize_embeddings,
        embedding_dtype,
        embedding_provider,
        embedding_dimension,
    )
    model_dir = os.path.dirname(first_array_path)
    os.makedirs(model_dir, exist_ok=True)
    manifest_path = os.path.join(model_dir, 'preparation_manifest.json')
    if selected_players and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                previous_manifest = json.load(f)
            previous_players = previous_manifest.get('players', {})
            if isinstance(previous_players, dict):
                player_summaries = {**previous_players, **player_summaries}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'schema_version': 1,
            'algorithm': 'twin',
            'dataset': 'rally',
            'provider': embedding_provider,
            'model_name': embedding_model_name,
            'revision': embedding_revision,
            'dimension': embedding_dimension,
            'normalize_embeddings': bool(normalize_embeddings),
            'dtype': embedding_dtype,
            'players': player_summaries,
        }, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f"Twin embeddings ready: {built} built, {skipped} already valid")
    print(f"Embedding caches saved to: {cache_dir}")
    return {'built': built, 'skipped': skipped}


def prepare_rallynetv2_data(df, split_content, player2index, output_dir):
    """
    Prepare data for RallyNet.
    
    Note: This function is kept for compatibility but may need updates
    based on specific RallyNet requirements.
    """
    player_trajectories_data = []
    match_ids_trajectories = []
    player_trajectory_data = []
    player_state_action_pair = []
    player_state_list = []
    player_action_list = []
    train_match_ids = split_content.get('train', {})
    index2player = { v: k for k, v in player2index.items() }
    player_match_ids_temp = []

    for index, row in df.iterrows():
        current_state = [row[col] for col in state_col]
        current_action = [row[col] for col in action_col]
        
        player_state_action_pair = current_state + current_action
        player_trajectory_data.append(player_state_action_pair)
        player_match_ids_temp.append(row["match_id"])

        nan_flag = np.isnan(row['server'])
        if row['server'] == 3 or nan_flag:
            player_trajectories_data.append(player_trajectory_data.copy())
            match_ids_trajectories.append(player_match_ids_temp.copy())
            player_trajectory_data, player_match_ids_temp = [], []  # 清空以供下一個軌跡使用

    player_masks_data = []

    for rally_idx, rally in enumerate(player_trajectories_data):

        rally_mask = []
        current_rally_match_ids = match_ids_trajectories[rally_idx]

        for ball_idx, ball in enumerate(rally):
            
            player_id = int(ball[17])
            player_name = index2player[player_id]
            match_id = int(current_rally_match_ids[ball_idx])

            player_train_matches = train_match_ids.get(player_name, [])
            if match_id in player_train_matches:
                rally_mask.append([1] * len(ball))
            else:
                rally_mask.append([0] * len(ball))

        player_masks_data.append(rally_mask)


    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle
    traj_path = os.path.join(output_dir, "train_total_traj.pkl")
    mask_path = os.path.join(output_dir, "train_total_mask.pkl")
    
    with open(traj_path, 'wb') as f:
        pickle.dump(player_trajectories_data, f)
    with open(mask_path, 'wb') as f:
        pickle.dump(player_masks_data, f)

    print(f"RallyNetV2 data saved to {output_dir}")
    print(f"  - train_total_traj.pkl: {len(player_trajectories_data)} samples")
    print(f"  - train_total_mask.pkl: {len(player_masks_data)} samples")

    ##########################################

    player_rallies = []
    opponent_rallies = []
    player_masks = []
    opponent_masks = []

    terminate_ball = [0]*23
    p_score = 0
    o_score = 0 

    for rally_idx, rally in enumerate(player_trajectories_data):
        rally_mask = player_masks_data[rally_idx]

        player_rally = []
        opponent_rally = []
        player_rally_mask = []
        opponent_rally_mask = []

        if len(rally) >= 2:
            player_num = rally[0][17]
            opponent_num = rally[1][17]
        else:
            player_num = rally[0][17]
            opponent_num = -1
            opponent_rally.append(terminate_ball.copy())
            opponent_rally_mask.append([0]*23)

        for ball_idx, ball in enumerate(rally):
            mask_value = rally_mask[ball_idx]
            if ball[17] == player_num:
                if np.isnan(ball[2]):
                    ball[2] = p_score
                    ball[3] = o_score
                    ball[4] = p_score - o_score
                else:
                    p_score = ball[2]
                    o_score = ball[3]

                player_rally.append(ball[:23].copy())
                player_rally_mask.append(mask_value.copy())
            elif ball[17] == opponent_num:
                if np.isnan(ball[2]):
                    ball[2] = o_score
                    ball[3] = p_score
                    ball[4] = o_score - p_score
                else:
                    p_score = ball[2]
                    o_score = ball[3]
                    ball[2] = o_score
                    ball[3] = p_score
                    ball[4] = o_score - p_score

                opponent_rally.append(ball[:23].copy())
                opponent_rally_mask.append(mask_value.copy())

        if len(player_rally) != 0:
            player_rallies.append(player_rally)
            player_masks.append(player_rally_mask)
        else:
            print("fatal error player")

        if len(opponent_rally) != 0:
            opponent_rallies.append(opponent_rally)
            opponent_masks.append(opponent_rally_mask)
        else:
            print("fatal error opponent")
            print(rally)
    
    # Save as pickle
    player_traj_path = os.path.join(output_dir, "train_player_traj.pkl")
    player_mask_path = os.path.join(output_dir, "train_player_mask.pkl")
    opponent_traj_path = os.path.join(output_dir, "train_opponent_traj.pkl")
    opponent_mask_path = os.path.join(output_dir, "train_opponent_mask.pkl")
    
    with open(player_traj_path, 'wb') as f:
        pickle.dump(player_rallies, f)
    with open(player_mask_path, 'wb') as f:
        pickle.dump(player_masks, f)
    with open(opponent_traj_path, 'wb') as f:
        pickle.dump(opponent_rallies, f)
    with open(opponent_mask_path, 'wb') as f:
        pickle.dump(opponent_masks, f)

    print(f"  - train_player_traj.pkl: {len(player_rallies)} samples")
    print(f"  - train_player_mask.pkl: {len(player_masks)} samples")
    print(f"  - train_opponent_traj.pkl: {len(opponent_rallies)} samples")
    print(f"  - train_opponent_mask.pkl: {len(opponent_masks)} samples")
