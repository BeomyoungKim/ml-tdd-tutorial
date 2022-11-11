import torch

from src.models.evaluator import Evaluator


def test_evaluate():
    evaluator = Evaluator()
    evaluator.add_pr_curve_tensorboard()
