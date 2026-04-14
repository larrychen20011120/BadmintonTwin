area_groups = {
    "Deep Left Corner": [32],
    "Deep Right Corner": [28],
    "Rear Left Side": [1, 5],
    "Rear Right Side": [4, 8],
    "Rear Center": [2, 3, 6, 7],
    "Mid Left Side": [9, 13],
    "Mid Right Side": [12, 16],
    "Mid Center": [10, 11, 14, 15],
    "Front Left Side": [20, 24],
    "Front Right Side": [17, 21],
    "Front Center": [18, 19, 22, 23],
    "Front Left Side Line": [29],
    "Front Right Side Line": [25],
    "Mid Left Side Line": [30],
    "Mid Right Side Line": [26],
    "Rear Left Side Line": [31],
    "Rear Right Side Line": [27],
    "Out of Bounds": [33]
}

EN_SHOT2INDEX = {
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
}
code_groups = {f"Z{i+1:02d}": ids for i, ids in enumerate(area_groups.values())}

INDEX2EN_SHOT = {
    v: k for k, v in EN_SHOT2INDEX.items()
}
LANG_AREA_MAPPING = {i: name for name, ids in area_groups.items() for i in ids}
CODE_AREA_MAPPING = {i: name for name, ids in code_groups.items() for i in ids}