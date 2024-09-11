from __future__ import annotations

import cupy as cp  # type: ignore


def compute_accuracy(model, noise: float = 0.0, deviations: float = 0.0) -> float:
    _, n_cells, n_samples = model.trajectories.shape

    epsilon = (
        noise * n_cells + (noise * (1 - noise) * n_cells) ** (1 / 2) * deviations
    ) // 1

    midpoint = n_cells // 2 + 1
    start_above = model.trajectories[0, :, :].sum(axis=0) >= midpoint
    start_below = model.trajectories[0, :, :].sum(axis=0) < midpoint
    end_above = model.trajectories[-1, :, :].sum(axis=0) >= (n_cells - epsilon)
    end_below = model.trajectories[-1, :, :].sum(axis=0) <= epsilon

    correct = (start_above & end_above) | (start_below & end_below)

    return correct.sum() / n_samples