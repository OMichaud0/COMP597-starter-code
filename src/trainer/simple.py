from typing import Any, Dict, Optional
import src.trainer.base as base
import src.config as config
import src.trainer.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import logging

logger = logging.getLogger(__name__)

class SimpleTrainer(base.Trainer):
    """Trainer for a simple iteration.

    This trainer implements a simple iteration step for a single device.

    Parameters
    ----------
    loader
        A PyTorch dataloader that will be used to obtain the data at each step.
    model
        The model to train.
    optimizer
        The PyTorch optimizer used to update the models weights.
    lr_scheduler
        A learning rate scheduler configured to work with the provided 
        optimizer.
    device
        The device on which the input batches will be moved.
    stats
        An object to gather statistics during training.

    Attributes
    ----------
    loader : torch.utils.data.DataLoader
        The object used to load data during training.
    model : torch.nn.Module
        The model to train as provided to the constructor.
    optimizer : torch.optim.Optimizer
        The optimizer used during training as provided to the constructor.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler used during training as provided to the 
        constructor.
    device : torch.device
        The device used to move the input batches as provided to the 
        constructor.
    stats : src.trainer.stats.TrainerStats
        The `TrainerStats` object used to gather statistics.

    """

    def __init__(self,
                 loader : data.DataLoader,
                 model : nn.Module,
                 optimizer : optim.Optimizer,
                 lr_scheduler : optim.lr_scheduler.LRScheduler,
                 device : torch.device,
                 stats : stats.TrainerStats,
                 conf: Optional[config.Config] = None):
        super().__init__(model, loader, device, stats)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # TODO remove conf as it is unused.
        self.conf = conf
        self._last_grad_norm = 0.0

    def checkpoint_dict(self, i: int) -> Dict[str, Any]:
        super_dict = super().checkpoint_dict(i)
        super_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        super_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return super_dict

    def _compute_grad_norm(self) -> float:
        """Compute the L2 norm of all gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        return math.sqrt(total_norm)

    def _compute_weight_norm(self) -> float:
        """Compute the L2 norm of all weights."""
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2).item()
            total_norm += param_norm ** 2
        return math.sqrt(total_norm)

    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        self.optimizer.zero_grad() #Zero the gradients
        outputs = self.model(**batch, **model_kwargs)
        return outputs.loss

    def backward(self, i: int, loss: torch.Tensor) -> None:
        loss.backward()
        self._last_grad_norm = self._compute_grad_norm()
        has_nan_grad = False
        has_inf_grad = False
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    has_nan_grad = True
                if torch.isinf(p.grad).any():
                    has_inf_grad = True
        if has_nan_grad or has_inf_grad or math.isnan(self._last_grad_norm) or math.isinf(self._last_grad_norm):
            logger.warning(f"Step {i}: ALERT - Gradient issue detected!")
            logger.warning(f"  Grad norm: {self._last_grad_norm}, NaN: {has_nan_grad}, Inf: {has_inf_grad}")

    def optimizer_step(self, i: int) -> None:
        weight_norm_before = self._compute_weight_norm()
        self.optimizer.step()
        self.lr_scheduler.step()
        weight_norm_after = self._compute_weight_norm()

        # Check for NaN/Inf in weights
        has_nan_weight = False
        has_inf_weight = False
        for p in self.model.parameters():
            if torch.isnan(p.data).any():
                has_nan_weight = True
            if torch.isinf(p.data).any():
                has_inf_weight = True

        if has_nan_weight or has_inf_weight or math.isnan(weight_norm_after) or math.isinf(weight_norm_after):
            logger.warning(f"Step {i}: ALERT - Weight issue detected after optimizer step!")
            logger.warning(f"  Weight norm before: {weight_norm_before:.4e}, after: {weight_norm_after:.4e}")
            logger.warning(f"  NaN: {has_nan_weight}, Inf: {has_inf_weight}")
        elif i < 5 or i % 10 == 0:  # Log first 5 steps and then every 10
            logger.info(f"Step {i}: Weight norm {weight_norm_before:.4e} -> {weight_norm_after:.4e}, Grad norm {self._last_grad_norm:.4e}")
