from Environment import BadmintonEnv
from Agent.BC import BC
from Agent.RallyNet import RallyNet
import pickle
import torch
import numpy as np

def main(player1,player2):
    # ================ Load real player names list =============== #
    with open('./input_data/target_players_ids.pkl', 'rb') as f:
        target_players = pickle.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================ Specify the player name and load your agent (we provide the Behavior Cloning Model here) =============== #
    # player_name = player1
    # player_id = target_players.index(player_name)
    # player_agent = BC(32, 128, len(target_players), 12, player_id)
    # player_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/BC_weight.pth", weights_only=True))
    # player_agent.eval()

    # ================ Or use our proposed RallyNetv2 as player  =============== #
    player_name = player1
    player_id = target_players.index(player_name)
    with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
        hyperparameters = pickle.load(f)
    player_agent = RallyNet(
            data_size= hyperparameters['data_size'], 
            latent_size=hyperparameters['latent_size'],
            context_size=hyperparameters['context_size'],
            hidden_size=hyperparameters['hidden_size'],
            player_ids_len = hyperparameters['player_ids_len'],
            target_players = hyperparameters['target_players'],
            shot_type_len = hyperparameters['shot_type_len'],
            ts = hyperparameters['ts'],
            id = player_id,
            device = device,
            player_name = player1,
        ).to(device)
    player_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/weight/Ralleynet/gen_e_1000.trc", map_location = device, weights_only=True),strict = False)


    # ================ Specify the real opponent AI using RallyNet (the state-of-the-art imitation learning model) =============== #
    opponent_name = player2
    with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
        hyperparameters = pickle.load(f)
    opponent_id = target_players.index(opponent_name)
    opponent_agent = RallyNet(
            data_size= hyperparameters['data_size'], 
            latent_size=hyperparameters['latent_size'],
            context_size=hyperparameters['context_size'],
            hidden_size=hyperparameters['hidden_size'],
            player_ids_len = hyperparameters['player_ids_len'],
            target_players = hyperparameters['target_players'],
            shot_type_len = hyperparameters['shot_type_len'],
            ts = hyperparameters['ts'],
            id = opponent_id,
            device = device,
            player_name = player2,
        ).to(device)
    opponent_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/weight/Ralleynet/gen_e_1000.trc", map_location = device, weights_only=True),strict = False)

    # ================ Specify rally length =============== #
    rallies = 200

    # ================ Construct the environment for interaction =============== #
    env = BadmintonEnv(player_agent, opponent_agent, player_name, opponent_name, rallies, 
                    is_match = True, is_constraint = True, is_have_serve_state = False, filepath = './output_data/output_game_sim_K_V.csv')

    # ================ Competition starts =============== #
    total_rewards = []
    total_rounds = []
    for rally in range(1, rallies + 1):
        print("rally :", rally)
        states, info, done, launch = env.reset()
        rally_reward = 0
        while not done :
            
            action = player_agent.action(states, info, launch)
            states, reward, info, done, launch = env.step(action, launch)
            rally_reward += reward
            if info['round'][-1] >= 59:
                # reset = 1
                break
            
        print("score: ", info['env_score'])
        print("round: ", info['round'][-1]-1)
        print("reward: ", rally_reward)
        total_rewards.append(rally_reward)
        total_rounds.append(info['round'][-1]-1)
        print()
    # ================ Close the environment and save the result =============== #
    print(f"Average Reward: {np.mean(total_rewards)} / Average Rounds: {np.mean(total_rounds)}")
    env.close()

    return list(env.output.to_dict(orient="records"))

if __name__ == "__main__":
    main('Viktor AXELSEN','Kento MOMOTA')