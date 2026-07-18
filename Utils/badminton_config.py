area_groups = {
    "Deep Left Corner": [32],
    "Deep Right Corner": [28],
    "Rear Left Side": [1, 5],
    "Rear Right Side": [4, 8],
    "Rear Center": [2, 3, 6, 7],
    "Mid Left Side": [9, 13],
    "Mid Right Side": [12, 16],
    "Mid Center": [10, 11, 14, 15],
    "Front Right Side": [20, 24],
    "Front Left Side": [17, 21],
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
    "Missed shot": 11,
}
code_groups = {f"Z{i+1:02d}": ids for i, ids in enumerate(area_groups.values())}

INDEX2EN_SHOT = {
    v: k for k, v in EN_SHOT2INDEX.items()
}
LANG_AREA_MAPPING = {i: name for name, ids in area_groups.items() for i in ids}
CODE_AREA_MAPPING = {i: name for name, ids in code_groups.items() for i in ids}


def _normalize_shot_name(value):
    return " ".join(str(value).strip().replace("_", " ").replace("-", " ").split()).lower()


_SHOT_NAME_ALIASES = {
    _normalize_shot_name(name): index for name, index in EN_SHOT2INDEX.items()
}
_SHOT_NAME_ALIASES.update({
    "serve short": 1,
    "short serve": 1,
    "serve long": 10,
    "long serve": 10,
    "smash defense": 5,
    "smash defence": 5,
    "net shot": 7,
    "net": 7,
    "push": 3,
    "push shot": 3,
    "missed": 11,
    "missed shot": 11,
    "miss": 11,
})


def shot_type_to_index(value):
    """Convert a shot type id or English shot name to its internal 1-11 id."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Invalid shot type: {value}")

    if isinstance(value, (int, float)):
        index = int(value)
        if float(value) == index and 1 <= index <= 11:
            return index
        raise ValueError(f"Invalid shot type: {value}")

    text = str(value).strip()
    if not text:
        raise ValueError("Invalid empty shot type")

    if text.isdigit():
        index = int(text)
        if 1 <= index <= 11:
            return index

    try:
        numeric_value = float(text)
        index = int(numeric_value)
        if numeric_value == index and 1 <= index <= 11:
            return index
    except ValueError:
        pass

    normalized = _normalize_shot_name(text)
    if normalized in _SHOT_NAME_ALIASES:
        return _SHOT_NAME_ALIASES[normalized]

    if "(" in normalized and normalized.endswith(")"):
        normalized = normalized[:normalized.rfind("(")].strip()
        if normalized in _SHOT_NAME_ALIASES:
            return _SHOT_NAME_ALIASES[normalized]

    raise ValueError(f"Unknown shot type: {value}")
