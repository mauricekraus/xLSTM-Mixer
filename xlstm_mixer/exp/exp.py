import inspect
from pathlib import Path
from typing import Any, Literal
from lightning import LightningModule
import os
from matplotlib import pyplot as plt
import torch

from xlstm_mixer.models.base_model import BaseModel
from ..models.xLSTMTime import xLSTMTime
from ..models.xlstm_mixer import xLSTMMixer
from ..models.lib_models.PatchTST_fixed import PatchTST
from ..models.lib_models.TimesNet_fixed import TimesNet
from ..models.lib_models.TimeMixer_fixed import TimeMixer
from ..models.lib_models.TiDE_fixed import TiDE
from ..models.lib_models.iTransformer_fixed import iTransformer
from ..models.lstm import LSTM

from ..utils.torchmetric_ext import CappedMeanAbsolutePercentageError
from ..lit.enums import Task, ForecastingTaskOptions, TimeFreq
from torch import nn
import torchmetrics as tm

BatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

import numpy as np
def mape(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)

class ForecastingExp(LightningModule):

    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | xLSTMMixer | PatchTST | TimesNet | iTransformer| xLSTMTime | LSTM | TimeMixer,
        task: Task,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
        metrics: list[tm.Metric] | dict[str, tm.Metric] = [tm.MeanSquaredError()],
    ) -> None:
        super().__init__()
        # self.seq_len = seq_len
        # self.pred_len = pred_len
        # self.enc_in = enc_in
        self.task_options = task_options
        self.criterion = criterion

        self.train_metrics = tm.MetricCollection(metrics, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        # Initialize the model with the provided class and kwargs
        self.model = architecture
        self.model.task = task

        self.pred_len = self.model.pred_len
        self.seq_len = self.model.seq_len
        self.enc_in = self.model.enc_in

        self.task = task

        self.val_outputs = []
        self.val_targets = []
        self.test_outputs = []
        self.test_targets = []
        self.mem_tokens = []
        # if self.model.mem_tokens is not None:
        #     for i in range(len(self.model.mem_tokens)):
        #         setattr(self, f"tok_content_token_{i}", [])

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

        if self.task == Task.SHORT_TERM_FORECAST and self.trainer.datamodule.data == "PEMS":
            B, T, C = outputs.shape
            outputs = self.trainer.datamodule.val_dataset.inverse_transform(outputs.reshape(-1, C).detach().cpu().numpy()).reshape(B, T, C)
            batch_y = self.trainer.datamodule.val_dataset.inverse_transform(batch_y.reshape(-1, C).detach().cpu().numpy()).reshape(B, T, C)
            self.val_outputs.append(outputs)
            self.val_targets.append(batch_y)
            outputs = torch.from_numpy(outputs).to(batch_x.device)
            batch_y = torch.from_numpy(batch_y).to(batch_x.device)

        self.log_dict(
            self.val_metrics(outputs, batch_y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        # self.mem_tokens.append(self.model.mem_tokens)

        return loss
    
    def plot_mem_tokens(self, tok_path , mem_tokens, name, idx):

        upscaled_mem_tokens = np.tile(mem_tokens.cpu().view(1,-1), (4,1)) # Repeat the memory token 10 times for visual clarity
        upscaled_mem_tokens = upscaled_mem_tokens.transpose(0, 1)  # Transpose to have the memory token on the x-axis
        getattr(self, f"tok_content_token_{idx}").append(upscaled_mem_tokens)


        if self.trainer.current_epoch == (self.trainer.max_epochs - 1):
            tok_content = getattr(self, f"tok_content_token_{idx}")
            tok_content = np.concatenate(tok_content, axis=0)
            # tok_content=tok_content
            np.save(tok_path / f"mem_tokens_{name}_{self.trainer.current_epoch}.npy", tok_content)
            # Plot the upscaled memory token as a heatmap
            plt.figure(figsize=(8, 6))
            plt.title("Upscaled Memory Token Heatmap")
            plt.imshow(tok_content, cmap='coolwarm', aspect='auto')
            plt.colorbar(label="Value")
            # color bar scale
            plt.clim(-0.35, 0.35)
            plt.xlabel("Hidden Dimension")
            plt.ylabel("Epochs")

            
                
            plt.savefig(tok_path / f"mem_tokens_{name}_{self.trainer.current_epoch}.png")

    
    def on_validation_epoch_end(self):
        if self.task == Task.SHORT_TERM_FORECAST and self.trainer.datamodule.data == "PEMS":
            val_outputs = np.concatenate(self.val_outputs, axis=0)
            val_targets = np.concatenate(self.val_targets, axis=0)
            mape_score = mape(val_outputs, val_targets)
            mape_score  = torch.tensor(mape_score)
            self.log("val/MeanAbsolutePercentageError", mape_score, prog_bar=True, on_epoch=True, on_step=False)
            self.val_outputs = []
            self.val_targets = []

        # if self.model.mem_tokens is not None:
        #     tok_path = Path("res") / f"tok_{self.model.mem_tokens.shape[0]}"
        #     tok_path.mkdir(parents=True, exist_ok=True)
        #     for i in range(self.model.mem_tokens.shape[0]):
        #         self.plot_mem_tokens(tok_path, self.mem_tokens[0][i], f"tok{i}", i)
        

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

        if self.task == Task.SHORT_TERM_FORECAST and self.trainer.datamodule.data == "PEMS":
            B, T, C = outputs.shape
            outputs = self.trainer.datamodule.test_dataset.inverse_transform(outputs.reshape(-1, C).detach().cpu().numpy()).reshape(B, T, C)
            batch_y = self.trainer.datamodule.test_dataset.inverse_transform(batch_y.reshape(-1, C).detach().cpu().numpy()).reshape(B, T, C)
            self.test_outputs.append(outputs)
            self.test_targets.append(batch_y)
            outputs = torch.from_numpy(outputs).to(batch_x.device)
            batch_y = torch.from_numpy(batch_y).to(batch_x.device)

        self.log_dict(
            self.test_metrics(outputs, batch_y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
    
    def on_test_epoch_end(self):
        if self.task == Task.SHORT_TERM_FORECAST and self.trainer.datamodule.data == "PEMS":
            test_outputs = np.concatenate(self.test_outputs, axis=0)
            test_targets = np.concatenate(self.test_targets, axis=0)
            mape_score = mape(test_outputs, test_targets)
            mape_score  = torch.tensor(mape_score)
            self.log("test/MeanAbsolutePercentageError", mape_score, prog_bar=True, on_epoch=True, on_step=False)
            self.test_outputs = []
            self.test_targets = []


class LongTermForecastingExp(ForecastingExp):

    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | xLSTMMixer,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
    ) -> None:
        super().__init__(
            criterion,
            architecture,
            Task.LONG_TERM_FORECAST,
            task_options,
            [tm.MeanAbsoluteError(), tm.MeanSquaredError()],
        )


class ShortTermForecastingExp(ForecastingExp):

    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | xLSTMMixer | LSTM | xLSTMTime,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
    ) -> None:
        super().__init__(
            criterion,
            architecture,
            Task.SHORT_TERM_FORECAST,
            task_options,
            {
                "MeanAbsoluteError": tm.MeanAbsoluteError(),
                # "MeanSquaredError": tm.MeanSquaredError(),
                "RootMeanSquaredError": tm.MeanSquaredError(squared=False),
                # "MeanAbsolutePercentageError": CappedMeanAbsolutePercentageError(),
                "SymmetricMeanAbsolutePercentageError": tm.SymmetricMeanAbsolutePercentageError(),
            },
        )


