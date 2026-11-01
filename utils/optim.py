import torch.optim as optim

from functools import partial

from utils.schedulers.delayed_sched import DelayedCosineAnnealingLR
from utils.schedulers.cosine_annealing_with_warmup import (
    CosineAnnealingWarmUpRestarts
)


def init_optimizer(
    optimizer_type,
    lr,
    weight_decay=0.0,
):
    """
    Returns an optimizer factory.

    Example:
        optimizer_f = init_optimizer(
            "adam",
            lr=1e-4,
            weight_decay=1e-5
        )

        optimizer = optimizer_f(
            model.parameters()
        )
    """

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":

        return partial(
            optim.Adam,
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "adamw":

        return partial(
            optim.AdamW,
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "sgd":

        return partial(
            optim.SGD,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    else:

        raise ValueError(
            f"Unknown optimizer: {optimizer_type}"
        )


def init_scheduler(
    scheduler_type,
    lr,
    epochs,
    warmup=3,
):
    """
    Returns a scheduler factory.

    Scheduler is created later using:

        scheduler = scheduler_f(optimizer)
    """

    if scheduler_type is None:

        return None

    scheduler_type = scheduler_type.lower()

    if scheduler_type in {
        "none",
        "constant",
    }:

        return None

    elif scheduler_type == "cosine":

        return partial(
            optim.lr_scheduler.CosineAnnealingLR,
            T_max=epochs,
        )

    elif scheduler_type == "cosine_warmup":

        return partial(
            CosineAnnealingWarmUpRestarts,
            T_0=epochs,
            T_up=warmup,
        )

    elif scheduler_type == "cosine_delayed":

        return partial(
            DelayedCosineAnnealingLR,
            delay_epochs=warmup,
            cosine_annealing_epochs=epochs,
        )

    elif scheduler_type == "step":

        return partial(
            optim.lr_scheduler.StepLR,
            step_size=max(epochs // 3, 1),
            gamma=0.1,
        )

    elif scheduler_type == "exp_decay":

        return partial(
            optim.lr_scheduler.ExponentialLR,
            gamma=0.95,
        )

    else:

        raise ValueError(
            f"Unknown scheduler: {scheduler_type}"
        )