from typing import List

import src.config as config
import src.trainer.stats.base as base
from src.trainer import stats as trainer_stats_module
import torch

trainer_stats_name = "composite"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    cfg = conf.trainer_stats_configs.composite
    names = [x.strip() for x in str(getattr(cfg, "components", "")).split(",") if x.strip()]
    if not names:
        raise ValueError("composite trainer stats requires at least one component")

    constructors = trainer_stats_module._TRAINER_STATS_CONSTRUCTORS
    children: List[base.TrainerStats] = []
    for name in names:
        if name == trainer_stats_name:
            raise ValueError("composite cannot include itself as a component")
        constructor_fn = constructors.get(name)
        if constructor_fn is None:
            raise ValueError(f"Unknown trainer stats component '{name}'")
        children.append(constructor_fn(conf, **kwargs))
    return CompositeTrainerStats(children)


class CompositeTrainerStats(base.TrainerStats):
    def __init__(self, components: List[base.TrainerStats]) -> None:
        super().__init__()
        self.components = components

    def start_train(self) -> None:
        for c in self.components:
            c.start_train()

    def stop_train(self) -> None:
        for c in self.components:
            c.stop_train()

    def start_step(self) -> None:
        for c in self.components:
            c.start_step()

    def stop_step(self) -> None:
        for c in self.components:
            c.stop_step()

    def start_forward(self) -> None:
        for c in self.components:
            c.start_forward()

    def stop_forward(self) -> None:
        for c in self.components:
            c.stop_forward()

    def log_loss(self, loss: torch.Tensor) -> None:
        for c in self.components:
            c.log_loss(loss)

    def start_backward(self) -> None:
        for c in self.components:
            c.start_backward()

    def stop_backward(self) -> None:
        for c in self.components:
            c.stop_backward()

    def start_optimizer_step(self) -> None:
        for c in self.components:
            c.start_optimizer_step()

    def stop_optimizer_step(self) -> None:
        for c in self.components:
            c.stop_optimizer_step()

    def start_save_checkpoint(self) -> None:
        for c in self.components:
            c.start_save_checkpoint()

    def stop_save_checkpoint(self) -> None:
        for c in self.components:
            c.stop_save_checkpoint()

    def log_step(self) -> None:
        for c in self.components:
            c.log_step()

    def log_stats(self) -> None:
        for c in self.components:
            c.log_stats()
