import os
import time
from typing import Any
import torch
import numpy as np
from torch import nn, optim
from pytorch_lightning import LightningModule

# from data_provider.m4 import M4Meta
from xlstm_mixer.models.base_model import BaseModel
from ..models.xlstm_mixer import xLSTMMixer
from ..lit.enums import Task, ForecastingTaskOptions
import torchmetrics as tm

BatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ShortTermForecastingExp(LightningModule):
    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | xLSTMMixer,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
    ):
        super().__init__()
        self.task_options = task_options
        self.model = architecture
        self.criterion = criterion

        self.train_metrics = tm.MetricCollection(
            [
                tm.MeanAbsolutePercentageError,
                tm.SymmetricMeanAbsolutePercentageError,
                tm.MeanSquaredError(squared=False),
                tm.MeanAbsoluteError(),
                tm.MeanSquaredError(),
            ],
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")
        self.model.task = Task.SHORT_TERM_FORECAST

        self.pred_len = self.model.pred_len
        self.seq_len = self.model.seq_len
        self.enc_in = self.model.enc_in

    # def _build_model(self):
    #     if self.args.data == "m4":
    #         self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
    #         self.args.seq_len = 2 * self.args.pred_len
    #         self.args.label_len = self.args.pred_len
    #         self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
    #     model = self.model_dict[self.args.model].Model(self.args).float()

    #     if self.args.use_multi_gpu and self.args.use_gpu:
    #         model = nn.DataParallel(model, device_ids=self.args.device_ids)
    #     return model

    def training_step(
        self, batch: BatchType
    ) -> torch.Tensor | os.Mapping[str, Any] | None:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_input = None
        outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

        f_dim = (
            -1
            if self.task_options == ForecastingTaskOptions.MULTIVARIATE_2_UNIVARIATE
            else 0
        )
        outputs = outputs[:, -self.pred_len :, f_dim:].contiguous()
        batch_y = batch_y[:, -self.pred_len :, f_dim:].contiguous()
        loss = self.criterion(outputs, batch_y)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)

        self.log_dict(
            self.train_metrics(outputs, batch_y),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def validation_step(
        self, batch: BatchType
    ) -> torch.Tensor | os.Mapping[str, Any] | None:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_input = None
        outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

        f_dim = (
            -1
            if self.task_options == ForecastingTaskOptions.MULTIVARIATE_2_UNIVARIATE
            else 0
        )
        outputs = outputs[:, -self.pred_len :, f_dim:].contiguous()
        batch_y = batch_y[:, -self.pred_len :, f_dim:].contiguous()
        loss = self.criterion(outputs, batch_y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.log_dict(
            self.val_metrics(outputs, batch_y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def test_step(self, batch: BatchType) -> torch.Tensor | os.Mapping[str, Any] | None:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_input = None
        outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

        f_dim = (
            -1
            if self.task_options == ForecastingTaskOptions.MULTIVARIATE_2_UNIVARIATE
            else 0
        )
        outputs = outputs[:, -self.pred_len :, f_dim:].contiguous()
        batch_y = batch_y[:, -self.pred_len :, f_dim:].contiguous()

        self.log_dict(
            self.test_metrics(outputs, batch_y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
