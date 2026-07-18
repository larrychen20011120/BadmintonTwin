import unittest

import torch

from Agent.bc import BC, gaussian_nll_loss, sample_from_gaussian
from Agent.shuttlenet import gaussian2d_nll, sample_gaussian2d


class BCGaussianDistributionTest(unittest.TestCase):
    def test_gaussian_heads_predict_state_dependent_distribution(self):
        model = BC(32, 128, 43, 12, player_id=0, distribution='gaussian')
        land, shot, move = model(torch.zeros(4, 18))

        self.assertEqual(land.shape, (4, 5))
        self.assertEqual(shot.shape, (4, 12))
        self.assertEqual(move.shape, (4, 5))
        self.assertFalse(any('log_sigma' in name for name, _ in model.named_parameters()))

    def test_other_distribution_head_shapes_are_unchanged(self):
        state = torch.zeros(4, 18)
        deterministic = BC(32, 128, 43, 12, player_id=0, distribution='deterministic')
        gmm = BC(32, 128, 43, 12, player_id=0, distribution='gmm', gmm_num=5)

        self.assertEqual(deterministic(state)[0].shape, (4, 2))
        self.assertEqual(deterministic(state)[2].shape, (4, 2))
        self.assertEqual(gmm(state)[0].shape, (4, 25))
        self.assertEqual(gmm(state)[2].shape, (4, 25))

    def test_nll_matches_shuttlenet(self):
        pred = torch.tensor([
            [0.1, -0.2, -1.0, 0.5, 0.0],
            [-0.3, 0.4, -8.0, 8.0, 1.5],
        ])
        target = torch.tensor([[0.2, -0.4], [-0.1, 0.6]])

        for reduction in ('none', 'sum', 'mean'):
            actual = gaussian_nll_loss(pred, target, reduction=reduction)
            expected = gaussian2d_nll(pred, target, reduction=reduction)
            torch.testing.assert_close(actual, expected)

    def test_sampling_matches_shuttlenet(self):
        pred = torch.tensor([
            [0.1, -0.2, -1.0, 0.5, 0.0],
            [-0.3, 0.4, -8.0, 8.0, 1.5],
        ])

        torch.manual_seed(1234)
        actual = sample_from_gaussian(pred)
        torch.manual_seed(1234)
        expected = sample_gaussian2d(pred)

        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all(actual >= -1.0))
        self.assertTrue(torch.all(actual <= 1.0))

    def test_gaussian_helpers_reject_non_gaussian_heads(self):
        pred = torch.zeros(2, 2)
        target = torch.zeros(2, 2)

        with self.assertRaises(ValueError):
            gaussian_nll_loss(pred, target)
        with self.assertRaises(ValueError):
            sample_from_gaussian(pred)


if __name__ == '__main__':
    unittest.main()
