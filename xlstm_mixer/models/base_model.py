from torch import nn
import torch

from xlstm_mixer.lit.enums import Task


class BaseModel(nn.Module):

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        enc_in: int = 1, 
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self._task = (
            Task.LONG_TERM_FORECAST
        )  # default task will be overwritten by the parent module

    # Define the property
    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task: Task):
        self._task = task

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None):
        raise NotImplementedError

    def imputation(self, x_enc):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc):
        raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        match self.task:
            case Task.LONG_TERM_FORECAST | Task.SHORT_TERM_FORECAST:
                return self.forecast(x_enc, x_mark_enc)
            case _:
                raise NotImplementedError
