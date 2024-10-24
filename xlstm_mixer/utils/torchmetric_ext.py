import torch
from torchmetrics import Metric
from typing import Any, Tuple, Union
import numpy as np

# Custom update function to mimic the paper's logic
def _custom_mape_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.17e-06,
    threshold: float = 5.0,  # Paper-specific threshold
) -> Tuple[torch.Tensor, int]:
    """Update and return variables required to compute Mean Absolute Percentage Error with paper-specific logic.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        epsilon: Specifies the lower bound for target values. Any target value below epsilon
            is set to epsilon (avoids ZeroDivisionError).
        threshold: Threshold above which percentage error is set to 0 (as per paper).

    """

    # Compute absolute percentage error
    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)

    # Set percentage errors > threshold to 0
    abs_per_error = torch.where(abs_per_error > threshold, torch.tensor(0.0, device=abs_per_error.device), abs_per_error)

    sum_abs_per_error = torch.sum(abs_per_error)
    num_obs = target.numel()

    return sum_abs_per_error, num_obs


def _custom_mape_compute(sum_abs_per_error: torch.Tensor, num_obs: Union[int, torch.Tensor]) -> torch.Tensor:
    """Compute Mean Absolute Percentage Error (MAPE) based on the paper-specific logic.

    Args:
        sum_abs_per_error: Sum of absolute value of percentage errors over all observations
        num_obs: Number of predictions or observations
    """
    return sum_abs_per_error / num_obs


def custom_mean_absolute_percentage_error(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute custom mean absolute percentage error based on the paper's logic.

    Args:
        preds: Predicted values
        target: Ground truth values

    Return:
        Tensor with MAPE
    """
    sum_abs_per_error, num_obs = _custom_mape_update(preds, target)
    return _custom_mape_compute(sum_abs_per_error, num_obs)


class CappedMeanAbsolutePercentageError(Metric):
    """Compute custom Mean Absolute Percentage Error (MAPE) based on the paper's logic.

    This follows the formula:
    MAPE = 1/n * sum( |y - y_hat| / max(epsilon, |y|) ), but values where percentage error > 5 are set to 0.
    """
    
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_per_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        threshold: float = 5.0,  # Default threshold for ignoring large errors
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("sum_abs_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _custom_mape_update(preds, target, threshold=self.threshold)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> torch.Tensor:
        """Compute mean absolute percentage error over state."""
        return _custom_mape_compute(self.sum_abs_per_error, self.total)