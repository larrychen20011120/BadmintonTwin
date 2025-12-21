# Environment Usage and RalleyNetv2

## Test and Evaluate the Environment
To run the evironment
```
python BadmintonEnv/RunEnv.py
```
This will generate `output_data/output_game_1.csv`.

To align the outputdata of the environment as input of evaluation modules, run:
```
python Test_preprocess/Current_dataset/data_preprocessing/run_test.py
```
This will generate `output_data/reordered_csv1.csv`.

If you have different input path for run_test.py, please edit the input path in the end of the file Test_preprocess/Current_dataset/data_preprocessing/test_dataPreprocessing.py
```
if __name__ == '__main__':
    preprocess('output_data/output_game_1.csv')
```

And run the evaluation modules to get the results of them:
```
python Tactic_Evaluation/main.py
python Visualization_And_Win_Lose_Reason_Statistics/test.py 
python Shot_Evaluation/test.py 
```

For shot evaluation, cause it need two matches to run some functions, so if you want to run it with single match file, please annotate the code below at line 67~70 in `Shot_Evaluation/main.py`
```
    # for i in rallies:
    #     fn7.append(plot_shot_influence.main(player_list[0], match_id, set_num, i, df, df_model1, dest_folder))
    # for i in rallies:
    #     fn8.append(plot_shot_influence.main(player_list[1], match_id, set_num, i, df, df_model2, dest_folder))
```

## Set Up the Environment
Refer to the code in `BadmintonEnv/RunEnv.py` or `BadmintonEnv/Baseline/train.py`.

```
with open('./input_data/target_players_ids.pkl', 'rb') as f:
    target_players = pickle.load(f)
with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
    hyperparameters = pickle.load(f)

device = BaseParameters.device

opponent_id = target_players.index('Viktor AXELSEN')
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
    player_name = opponent_name
).to(device)

opponent_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/weight/Ralleynet/gen_e_1000.trc", map_location = device, weights_only=True),strict = False)

env = BadmintonEnv(None, opponent_agent, 'Viktor AXELSEN', opponent_name, 10, 
                is_match = True, is_constraint = is_constraint, is_have_serve_state = False, filepath = './output_data/output_game_1.csv')

return env

```


## Action & State Definitions
- `action[0]`: Agent's shot type. (2 serve types, 8 shot types ,and miss shot)
- `action[1]`: Agent's hitting position (can return `null`).
- `action[2]`: Landing position of the agent's shot (`-177.5 ~ 177.5, 0 ~ 480`).
- `action[3]`: Agent's movement position after hitting the shuttle (`-177.5 ~ 177.5, -480 ~ 0`).
- `state[0]`: Opponent's shot type. (2 serve types, 8 shot types ,and miss shot)
- `state[1]`: Agent's position when the opponent hits the shuttle (`-177.5 ~ 177.5, -480 ~ 0`).
- `state[2]`: Opponent's position when hitting the shuttle (`-177.5 ~ 177.5, 0 ~ 480`).
- `state[3]`: Landing position of the opponent's shot (`-177.5 ~ 177.5, -480 ~ 0`).

## RalleyNetv2
RalleyNet determines whether the next state exists and confirms whether a player has made an error. Then, multiple Behavioral Cloning (BC) models predict the player's behavior.

To imitate a specific player who has not been modeled yet, for example, to imitate Anthony Sinisuka GINTING, run
```
python BadmintonEnv/Agent/train_BC.py Anthony_Sinisuka_GINTING
```

Then, update the `player_name` when calling RalleyNetv2.

## Baseline Method
We test our benchmark with 3 baseline methods with hybrid action space
- **PDQN** (https://arxiv.org/abs/1810.06394)
- **Hybrid PPO** (https://arxiv.org/abs/1903.01344)
- **Hybrid SAC** (https://arxiv.org/abs/1912.11077)

All of the detailed implementations can be found under `BadmintonEnv/Baseline/Agent` directory. It includes how agent updates it parameters, interact with environment, do the exploration, and so on.

### Model Structure
Our RL model structures are defined in `BadmintonEnv/Baseline/Net/Net.py` with only **linear** layer is needed.


To make our model learn to capture the difference between launch and non-launch, all our model **input size is 8**, which is the combination of observations from the environment and launch.

### Hyper-parameters
Our hyper-parameters of each method is declared in the `BadmintonEnv/Baseline/config.py`. For each baseline, just modify the corresponding class.

### Training

Our training script is written in the `BadmintonEnv/Baseline/train.py`. There are 4 arguments: **opponent**, **agent**, **constraint** and **log**. For the detailed definition, you can view the following table.

| Argument Name  | Argument Definition | Argument Type |
| :-------------: | :-------------: | :-------------: |
| opponent  | the name of the badminton players (currently support `Viktor AXELSEN` and `Kento MOMOTA`)  | string  |
| agent  | the RL agents with 3 choices (`PDQN`, `HPPO`, `HSAC`)  | string  |
| constraint  |  whether to apply constraint or not (just type `--constraint` or `--no-constraint`) | boolean  |
| log  | the log name of the training result and model snapshots (note: it will be saved under `log/YOUR_NAME`)  | string  |

---
example command script for training PDQN on opponent Viktor AXELSEN with environment constraint
```
python BadmintonEnv/Baseline/train.py --opponent 'Viktor AXELSEN' --agent PDQN --constraint --log PDQN-constraint-viktor
```
---

### Evaluation
We evaluate our models in different timestamps to check the improvement of the agent. Just run the `BadmintonEnv/Baseline/eval.py` with the path of the weight.  


---
example command script for evaluating PDQN on opponent Viktor AXELSEN with environment constraint
```
python BadmintonEnv/Baseline/eval.py --opponent 'Viktor AXELSEN' --agent PDQN --constraint --log test --weight "YOUR_WEIGHT_PATH"
```
---