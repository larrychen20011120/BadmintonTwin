import sys
import os
import argparse

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "BadmintonEnv"))
sys.path.append(project_path)
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Environment import BadmintonEnv
from Agent.RallyNet import RallyNet
import pickle
import torch
from tqdm import tqdm
from BadmintonGPT.agent import TwoStageLLMAgent
import yaml
import pandas as pd
import numpy as np

# def make_env(filepath, player_name="Kento MOMOTA", opponent_name='Viktor AXELSEN', is_constraint=False, n_rallies=200):

#     with open('./input_data/target_players_ids.pkl', 'rb') as f:
#         target_players = pickle.load(f)
#     with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
#         hyperparameters = pickle.load(f)

#     device = "cuda"

#     opponent_id = target_players.index(opponent_name)
#     opponent_agent = RallyNet(
#         data_size= hyperparameters['data_size'], 
#         latent_size=hyperparameters['latent_size'],
#         context_size=hyperparameters['context_size'],
#         hidden_size=hyperparameters['hidden_size'],
#         player_ids_len = hyperparameters['player_ids_len'],
#         target_players = hyperparameters['target_players'],
#         shot_type_len = hyperparameters['shot_type_len'],
#         ts = hyperparameters['ts'],
#         id = opponent_id,
#         device = device,
#         player_name = opponent_name
#     ).to(device)
    
#     opponent_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/weight/Ralleynet/gen_e_1000.trc", map_location = device, weights_only=True),strict = False)

#     env = BadmintonEnv(
#         None, opponent_agent, player_name, opponent_name, n_rallies, 
#         is_match = True, is_constraint = is_constraint, is_have_serve_state = False, 
#         filepath = filepath
#     )

#     return env

def make_env(filepath, opponent_agent, player_name, is_constraint=False, n_rallies=200):

    env = BadmintonEnv(
        None, opponent_agent, player_name, opponent_agent.player_name, n_rallies, 
        is_match = True, is_constraint = is_constraint, is_have_serve_state = False, 
        filepath = filepath
    )

    return env

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--player', type=str, default="Kento MOMOTA")
    parser.add_argument('--opponent', type=str, default='ALL', help="rally net type: default is BC from the whole dataset")
    parser.add_argument('--constraint', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--filename', type=str, help="the path to your model output", default="my_output_game")
    parser.add_argument('--situation', type=str, default="選手現在腳受傷，移動能力不如往常")
    parser.add_argument('--n_rallies', type=int, default=50, help="the number of rallies to play")

    args = parser.parse_args()

    with open("BadmintonGPT/openai_two_stage.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv("input_data/my_processed_data.csv")

    opponent_agent = TwoStageLLMAgent(
        player=args.opponent, df=df,
        model_name=config["experiment_configs"]["qwen"]["model_name"],
        api_key=config["experiment_configs"]["qwen"]["api_key"]
    )

    player_agent = TwoStageLLMAgent(
        player=args.player, df=df,
        model_name=config["experiment_configs"]["qwen"]["model_name"],
        api_key=config["experiment_configs"]["qwen"]["api_key"],
        situation=args.situation
    )

    env = make_env(
        filepath=os.path.join("output_data", args.filename) + ".csv",
        opponent_agent=opponent_agent, player_name=args.player,
        is_constraint=args.constraint,
        n_rallies=args.n_rallies
    )

    total_rewards, total_rounds = [], []
    for rally in tqdm(range(1, args.n_rallies + 1)):
        print("rally :", rally)
        states, info, done, launch = env.reset()
        # player_agent.reset_time()
        # opponent_agent.reset_time()
        
        rally_reward = 0

        while not done :
            
            action = player_agent.action(states, info, launch)
            ### ADD NEW CODE
            if action is None:
                reason="失誤"
            elif len(action) == 2:
                reason = action[1]
                action = action[0]
            ###
            states, reward, info, done, launch = env.step(action, launch, reason=reason)
            rally_reward += reward
            # if info['round'][-1] >= 59:
            #     break

        total_rewards.append(rally_reward)
        total_rounds.append(info['round'][-1]-1)
            
        print("score: ", info['env_score'])
        print("round: ", info['round'][-1]-1)
        print("reward: ", rally_reward)
        print()
    # ================ Close the environment and save the result =============== #
    env.close()
    print("Average Reward: ", np.mean(total_rewards))
    print("Average Rounds: ", np.mean(total_rounds))
    