from BC_model._model import SimpleNN as NN_move 
from BC_model.s_model import SimpleNN as NN_type
from BC_model.serve_model import SimpleNN as NN_serve 
from BC_model.shot_model import SimpleNN as NN_shot 

import os
import sys

if __name__ == "__main__":
    player = 'Anthony Sinisuka GINTING'
    gmm_c = 5

    if len(sys.argv) > 1:
        player = sys.argv[1]
        player = player.replace("_", " ")

    if not os.path.exists('BadmintonEnv/Agent/weight/'+player):
        os.makedirs('BadmintonEnv/Agent/weight/'+player)

    for i in range(8):
        model = NN_type(gmm_c)
        shot_type = i +2
        model.train_model('input_data/2_dataset_merge.csv', shot_type=shot_type,player = player)

    model = NN_serve(gmm_c)
    shot_type = 1
    model.train_model('input_data/Processed_dataset_merge3.csv', shot_type=shot_type,player = player)
    shot_type = 10
    model.train_model('input_data/Processed_dataset_merge3.csv', shot_type=shot_type,player = player)

    model = NN_shot()
    model.train_model('input_data/2_dataset_merge.csv',player = player)

    model = NN_move(gmm_c)
    model.train_model('input_data/2_dataset_merge.csv',player = player)