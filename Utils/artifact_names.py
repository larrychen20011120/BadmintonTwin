"""Stable collision-resistant names for player-specific artifacts."""

import hashlib
import re


def player_database_filename(player_name):
    name = str(player_name)
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') or 'player'
    digest = hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]
    return f'{slug}_{digest}_db.pkl'
