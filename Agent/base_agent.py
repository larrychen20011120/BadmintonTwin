from dtaidistance import dtw

class BaseAgent:
    def __init__(self, player_name, is_infernce=True):

        self.player_name = player_name
        # is_infernce       --> [-177.5~177.5, 0~-480]
        # not_is_inference  --> [-1~1. -1~1]
        self.is_infernce = is_infernce

        self.tw_to_en_type = {
            "發短球": "Serve short",
            "發長球": "Serve long",
            "長球": "Clear",
            "推撲球": "Push Shot",
            "殺球": "Smash",
            "接殺防守": "Smash Defence",
            "平球": "Drive",
            "接不到": "Missed shot",
            "網前球": "Net Shot",
            "挑球": "Lob",
            "切球": "Drop",
        }

        self.type_to_index = {
            "Serve short": 1,
            "Clear": 2,
            "Push Shot": 3,
            "Smash": 4,
            "Smash Defence": 5,
            "Drive": 6,
            "Net Shot": 7,
            "Lob": 8,
            "Drop": 9,
            "Serve long": 10,
            "Missed shot": 11,
        }
        self.index_to_type = {
            v: k for k, v in self.type_to_index.items()
        }

    def predict(self, state, info, launch):
        raise NotImplementedError("Subclasses must implement evaluate")

    def action(self, state, info, launch):
        normalized_action = self.predict(state, info, launch)
        return self.decode_action(normalized_action)
    

    def encode_state(self, states):
        # Normalize positions
        player_x = (states[1][0]/177.5)
        player_y = (states[1][1]/240+1)
        hit_x = (states[3][0]/177.5)
        hit_y = (states[3][1]/240+1)
        opponent_x = -(states[2][0]/177.5)
        opponent_y = -(states[2][1]/240-1)
        pre_type = states[0]
        return [pre_type, player_x, player_y, opponent_x, opponent_y, hit_x, hit_y]

    def decode_action(self, action, reason=""):
        output_shot, hit, output_land, output_move, dist = action

        output_land[0] = -float(output_land[0]) * 177.5
        output_land[1] = -240 * float(output_land[1]) + 240 
        output_move[0] = float(output_move[0]) * 177.5 
        output_move[1] = 240 * float(output_move[1]) - 240

        return (output_shot, hit, tuple(output_land), tuple(output_move), dist), reason