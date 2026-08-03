import unittest

from Utils.artifact_names import player_database_filename


class ArtifactNamesTest(unittest.TestCase):
    def test_player_database_names_are_stable_and_collision_resistant(self):
        self.assertEqual(
            player_database_filename('A.B'),
            player_database_filename('A.B'),
        )
        self.assertNotEqual(
            player_database_filename('A.B'),
            player_database_filename('AB'),
        )
        self.assertTrue(player_database_filename('Player A').endswith('_db.pkl'))


if __name__ == '__main__':
    unittest.main()
