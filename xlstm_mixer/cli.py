"""CLI interface for xlstm_mixer project.

"""

from copy import copy
import os
import sys

import torch
import wandb
from xlstm_mixer.lit.data import (
    TSLibDataModule,
)
from xlstm_mixer.lit.forecast_visualize_cb import ForecastVisualizeCallback
from xlstm_mixer.exp.exp import (
    LongTermForecastingExp,
    ShortTermForecastingExp,
    ForecastingExp,
)
import torch.optim as optim

from xlstm_mixer.cli_helper import LoggerSaveConfigCallback, TaskCLI
import sys


import optuna

from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging
from lightning import LightningModule
from lightning import Trainer
from multiprocessing import Pool


class PyTorchLightningPruningCallback(EarlyStopping):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:

        super().__init__(monitor=monitor, patience=15, mode="max")

        self._trial = trial

    def _process(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

    # NOTE (crcrpar): This method is called <0.8.0
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self._process(trainer, pl_module)

    # NOTE (crcrpar): This method is called >=0.8.0
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self._process(trainer, pl_module)


def fit_and_test(rest_args):
    cli = TaskCLI(
        ForecastingExp,
        TSLibDataModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        run=False,
        args=rest_args,
        save_config_callback=LoggerSaveConfigCallback,
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.callbacks = [
        cb
        for cb in cli.trainer.callbacks
        if not isinstance(cb, StochasticWeightAveraging)
    ]
    if cli.trainer.fast_dev_run:
        print("Fast dev run, skipping test")
        print(cli.trainer.logged_metrics)
        return
    
    cli.trainer.test(
        cli.model,
        cli.datamodule,
        ckpt_path=cli.trainer.checkpoint_callback.best_model_path,
    )
    wandb.finish()
    if "test/MeanSquaredError" not in cli.trainer.logged_metrics:
        mae = cli.trainer.logged_metrics["test/MeanAbsoluteError"].item()
        mape = cli.trainer.logged_metrics["test/MeanAbsolutePercentageError"].item()
        rmse = cli.trainer.logged_metrics["test/RootMeanSquaredError"].item()
        return {"mae": mae, "mape": mape, "rmse": rmse}

    mse_test = cli.trainer.logged_metrics["test/MeanSquaredError"].item()
    mae_test = cli.trainer.logged_metrics["test/MeanAbsoluteError"].item()

    return {"mse_test": mse_test, "mae_test": mae_test}


def tune(rest_args):
    cli = TaskCLI(
        ForecastingExp,
        TSLibDataModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        run=False,
        args=rest_args,
        save_config_callback=LoggerSaveConfigCallback,
        # trainer_defaults={"callbacks": [PyTorchLightningPruningCallback(trial, monitor="val/MeanSquaredError")]},
        save_config_kwargs={"overwrite": True},
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    if "val/MeanSquaredError" not in cli.trainer.logged_metrics:
        mae = cli.trainer.logged_metrics["val/MeanAbsoluteError"].item()
        mape = cli.trainer.logged_metrics["val/MeanAbsolutePercentageError"].item()
        rmse = cli.trainer.logged_metrics["val/RootMeanSquaredError"].item()
        smape = cli.trainer.logged_metrics["val/SymmetricMeanAbsolutePercentageError"].item()
    else:
        mse = cli.trainer.logged_metrics["val/MeanSquaredError"].item()
        mae = cli.trainer.logged_metrics["val/MeanAbsoluteError"].item()
    # if cli.trainer.checkpoint_callback.best_model_path is '':
        # cli.trainer.logged_metrics["test/MeanAbsoluteError"] = torch.tensor([-1])
        # cli.trainer.logged_metrics["test/MeanSquaredError"] = torch.tensor([-1])  
        # print("WARNING: No best model found. Test metrics are set to -1")
    # else:
    cli.trainer.test(
    cli.model,
    cli.datamodule,
    ckpt_path=cli.trainer.checkpoint_callback.best_model_path,
    )
    if "test/MeanSquaredError" not in cli.trainer.logged_metrics:
        mae_test = cli.trainer.logged_metrics["test/MeanAbsoluteError"].item()
        mape_test = cli.trainer.logged_metrics[
            "test/MeanAbsolutePercentageError"
        ].item()
        rmse_test = cli.trainer.logged_metrics["test/RootMeanSquaredError"].item()
        smape_test = cli.trainer.logged_metrics["test/SymmetricMeanAbsolutePercentageError"].item()
    else:
        mse_test = cli.trainer.logged_metrics["test/MeanSquaredError"].item()
        mae_test = cli.trainer.logged_metrics["test/MeanAbsoluteError"].item()
    wandb.finish()

    if "test/MeanSquaredError" not in cli.trainer.logged_metrics:
        return {
            "test/mae": mae_test,
            "val/mae": mae,
            "test/mape": mape_test,
            "val/mape": mape,
            "test/rmse": rmse_test,
            "val/rmse": rmse,
            "test/smape": smape_test,
            "val/smape": smape,
            "ckpt_path": cli.trainer.checkpoint_callback.best_model_path,
        }
    else:
        return {
            "test/mse": mse_test,
            "val/mse": mse,
            "val/mae": mae,
            "test/mae": mae_test,
            "ckpt_path": cli.trainer.checkpoint_callback.best_model_path,
        }


def run_task_cli():
    TaskCLI(
        ForecastingExp,
        TSLibDataModule,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=LoggerSaveConfigCallback,
    )


def helper(gpu1: bool, rest_args, gpu_start_idx: str, idx, seed):
    args = copy(rest_args)
    gpu = int(gpu_start_idx)
    if gpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if gpu1 else "1"
    elif gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" if gpu1 else "0"
    elif gpu == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2" if gpu1 else "3"
    elif gpu == 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = "4" if gpu1 else "5"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2" if gpu1 else "3"
    args[idx + 1] = seed
    res = fit_and_test(args)
    return res


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m xlstm_mixer` and `$ xlstm_mixer `.

    This is your program's entry point.
    """
    args = sys.argv

    if len(args) > 1:
        args = args[1:]
        match args:
            case ("fit+test", *rest_args):
                fit_and_test(rest_args)
                return
            case ("fit+test-multi", *rest_args):
                idx = rest_args.index("--seed_everything")

                gpu_idx = os.environ["CUDA_VISIBLE_DEVICES"]

                orig_seed = int(rest_args[idx + 1])
                cnt = 3
                with Pool(cnt) as p:
                    ress = p.starmap(
                        helper,
                        [
                            (i < 3, rest_args, gpu_idx, idx, str(orig_seed + i))
                            for i in range(cnt)
                        ],
                    )

                # ress = []
                # for i in range(3):
                #     rest_args[idx + 1] = str(orig_seed + i)
                #     res = fit_and_test(rest_args)
                #     ress.append(res)
                for i in range(cnt):
                    print(f"Seed 202{i+1}: {ress[i]}")
                print("Average:")
                print(
                    {
                        "mse_test": sum([res["test/mse"] for res in ress]) / cnt,
                        "mae_test": sum([res["test/mae"] for res in ress]) / cnt,
                    }
                )

                # rest_args[idx + 1]
                # fit_and_test(rest_args)
                return

            case ("tune", *rest_args):
                results = tune(rest_args)
                print(results)
                return

            case _:
                pass

    run_task_cli()


if __name__ == "__main__":
    main()
