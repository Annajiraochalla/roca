import matplotlib.pyplot as plt
from cubewalkers.model import Model
from matplotlib.figure import Figure

import roomyrobot.metrics as metrics


def plot_majority_classification_samples(
    model: Model,
    title: str = "",
    noise_for_tolerance: float = 0.0,
    deviations_for_tolerance: float = 0.0,
    walkers_to_plot: list[int] | None = None,
    font_size: int = 20,
) -> Figure:
    accuracy = metrics.compute_accuracy(
        model.trajectories,
        noise=noise_for_tolerance,
        deviations=deviations_for_tolerance,
    )
    if walkers_to_plot is None:
        walkers_to_plot = [0]

    aspect = model.n_time_steps // model.n_variables
    fig, axs = plt.subplots(  # type: ignore
        1,
        len(walkers_to_plot),
        constrained_layout=True,
        figsize=(3 * len(walkers_to_plot), 2 * aspect + 1),
        squeeze=False,
    )  # type: ignore
    fig.set_facecolor("white")
    fig.suptitle(  # type: ignore
        f"{title}\n"
        f"Samples: {model.n_walkers}, Noise: {noise_for_tolerance}, Accuracy: {accuracy}",
        fontsize=font_size,
    )
    for ind, ax in enumerate(axs.reshape(-1)):
        ax.imshow(
            model.trajectories[:, :, ind].get(), interpolation="none", cmap="viridis"
        )
        if (
            metrics.compute_accuracy(
                model.trajectories[:, :, ind : (ind + 1)],
                noise=noise_for_tolerance,
                deviations=deviations_for_tolerance,
            )
            >= 0.5
        ):
            ax.set_title("CORRECT", color="blue", fontsize=font_size)
        else:
            ax.set_title("INCORRECT", color="red", fontsize=font_size)
        ax.set_xlabel("cell number", fontsize=font_size)
        ax.set_ylabel("time step", fontsize=font_size)
    return fig
