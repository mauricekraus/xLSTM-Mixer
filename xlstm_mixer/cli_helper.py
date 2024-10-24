import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from jsonargparse import Namespace, lazy_instance
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    LearningRateMonitor,
)
import numpy as np
from torch.optim import Optimizer
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from xlstm_mixer.exp.exp import (
    LongTermForecastingExp,
)
import torch.optim as optim

from lightning.pytorch.loggers import WandbLogger, Logger
from lightning.fabric.utilities.cloud_io import get_filesystem
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from xlstm_mixer.models.base_model import BaseModel
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities.core.rank_zero import _warn
from functools import partial, update_wrapper
from types import MethodType
from lightning.pytorch.callbacks import BaseFinetuning


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, monitor: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.monitor = monitor


# LightningCLI requires the ReduceLROnPlateau defined here, thus it shouldn't accept the one from pytorch:
LRSchedulerTypeTuple = (LRScheduler, ReduceLROnPlateau)

def _class_path_from_class(class_type: Type) -> str:
    return class_type.__module__ + "." + class_type.__name__

def _global_add_class_path(
    class_type: Type, init_args: Optional[Union[Namespace, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    if isinstance(init_args, Namespace):
        init_args = init_args.as_dict()
    return {"class_path": _class_path_from_class(class_type), "init_args": init_args or {}}


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.

    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)

class WarmUpCosineAnnealingScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, gamma: float=0.98, 
                 constant_gamma_epochs: int=2, cosine_epochs: int=15, 
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.constant_gamma_epochs = constant_gamma_epochs
        self.cosine_annealing_epochs = cosine_epochs
        self.total_epochs = warmup_epochs + constant_gamma_epochs + cosine_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.print_dict = {"warmup": False, "constant": False, "cosine": False}
        
        super(WarmUpCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def print_swap(self, epoch):
        if epoch < self.warmup_epochs and not self.print_dict["warmup"]:
            self.print_dict["warmup"] = True
            rank_zero_info(f"Setting scheduler to Warm-Up")
        elif epoch >= self.warmup_epochs and epoch < self.warmup_epochs + self.constant_gamma_epochs and not self.print_dict["constant"]:
            self.print_dict["constant"] = True
            rank_zero_info(f"Swapping scheduler from Warm-Up to Constant Gamma at epoch {self.last_epoch + 1}")
        elif epoch >= self.warmup_epochs + self.constant_gamma_epochs and not self.print_dict["cosine"]:
            self.print_dict["cosine"] = True
            rank_zero_info(f"Swapping scheduler from Constant Gamma to Cosine Annealing at epoch {self.last_epoch + 1}")

    
    def get_lr(self):
        epoch = self.last_epoch + 1

        self.print_swap(epoch)
        
        if epoch < self.warmup_epochs:  # Warmup phase (linear increase)

            return [base_lr  for base_lr in self.base_lrs]
        
        elif epoch < self.warmup_epochs + self.constant_gamma_epochs:  # Constant gamma decay phase
            return [base_lr * self.gamma for base_lr in self.base_lrs]
        
        else:  # Cosine annealing phase
            cosine_epoch = epoch - (self.warmup_epochs + self.constant_gamma_epochs)

            ep = np.clip(cosine_epoch, 0, self.cosine_annealing_epochs-1)
            if cosine_epoch >= self.cosine_annealing_epochs:
                return [base_lr * self.gamma * 0.5 * (1 + math.cos(math.pi * ep / self.cosine_annealing_epochs)) 
                    for base_lr in self.base_lrs]
            return [base_lr * self.gamma * 0.5 * (1 + math.cos(math.pi * ep / self.cosine_annealing_epochs)) 
                    for base_lr in self.base_lrs]



# class WarmUpCosineAnnealingScheduler(lr_scheduler._LRScheduler):
#     def __init__(
#         self, optimizer, warmup_epochs, cosine_epochs, min_lr=0, last_epoch=-1
#     ):
#         self.warmup_epochs = warmup_epochs
#         self.cosine_epochs = cosine_epochs
#         self.min_lr = min_lr
#         self.total_epochs = warmup_epochs + cosine_epochs
#         self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=cosine_epochs, eta_min=min_lr
#         )

#         self.has_printed_transition = False
#         super(WarmUpCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < self.warmup_epochs:
#             # During warmup, the learning rate is constant
#             return [base_lr for base_lr in self.base_lrs]
#         else:
#              # Print a message once when transitioning to the cosine annealing phase
#             if not self.has_printed_transition:
#                 rank_zero_info(
#                     f"Swapping scheduler from Warm-Up to Cosine Annealing at epoch {self.last_epoch + 1}"
#                 )
#                 self.has_printed_transition = True
#             # After warmup, delegate to CosineAnnealingLR
#             self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
#             return self.cosine_scheduler.get_lr()

#     def step(self, epoch=None):
#         if self.last_epoch < self.warmup_epochs:
#             super(WarmUpCosineAnnealingScheduler, self).step(epoch)
#         else:
#             self.cosine_scheduler.step(
#                 epoch - self.warmup_epochs if epoch is not None else None
#             )
#             self.last_epoch = self.cosine_scheduler.last_epoch + self.warmup_epochs


# Exampl


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(
                self.config, skip_none=False, format="json"
            )  # Required for proper reproducibility
            trainer.logger.log_hyperparams(json.loads(config))

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if isinstance(trainer.logger, WandbLogger):
            run_id = trainer.logger.experiment.id
            log_dir = f"outputs/{run_id}"
            wandb_log_dir = os.path.join(log_dir, "wandb")
            trainer.logger._save_dir = wandb_log_dir
        else:
            run_id = "default"
            log_dir = f"outputs/{run_id}"

        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        if trainer.is_global_zero:
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

            # Also create the wandb log directory if using WandbLogger
            if isinstance(trainer.logger, WandbLogger):
                os.makedirs(wandb_log_dir, exist_ok=True)

            # Update checkpoint callback dirpath
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback.dirpath = os.path.join(log_dir, "checkpoints")

            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # Broadcast the already_saved status to ensure all ranks are in sync
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
     def __init__(self, unfreeze_at_epoch=4):
         super().__init__()
         self._unfreeze_at_epoch = unfreeze_at_epoch

     def freeze_before_training(self, pl_module):
         # freeze any module you want
         # Here, we are freezing `feature_extractor`
         self.freeze(pl_module.model.xlstm)
         rank_zero_info(
                    f"Freeze xlstm at {pl_module.trainer.current_epoch + 1}"
                )

     def finetune_function(self, pl_module, current_epoch, optimizer):
         # When `current_epoch` is 10, feature_extractor will start training.
         if current_epoch == self._unfreeze_at_epoch:
             self.unfreeze_and_add_param_group(
                 modules=pl_module.model.xlstm,
                 optimizer=optimizer,
                 train_bn=True,
             )
             rank_zero_info(
                    f"Free xlstm at {pl_module.trainer.current_epoch + 1}"
                )
             self.freeze(pl_module.model.tmixer)
             rank_zero_info(
                    f"Freeze timemixer at {pl_module.trainer.current_epoch + 1}"
                )



class TaskCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        # if parser.get_value("model.task") != Task.CLASSIFICATION:
        # parser.add_lightning_class_args(
        #     ForecastVisualizeCallback, "forecast_visualize_cb"
        # )

        parser.add_optimizer_args((optim.Adam, optim.RAdam))
        parser.add_lr_scheduler_args(
            (
                WarmUpCosineAnnealingScheduler,
                optim.lr_scheduler.StepLR,
                optim.lr_scheduler.CosineAnnealingLR,
                optim.lr_scheduler.ConstantLR,
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
                optim.lr_scheduler.SequentialLR,
            )
        )

        parser.add_lightning_class_args(RichModelSummary, "rich_model_summary")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")

        # torch.optim.lr_scheduler.CosineAnnealingLR
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    WandbLogger, project="xlstm_mixer", group="h-methods"
                ),
                "trainer.max_epochs": 30,
                "trainer.gradient_clip_val": 1.0,
                "seed_everything": 2021,
                "optimizer": {
                    "class_path": "torch.optim.Adam",
                    "init_args": {"lr": 0.01,},# "weight_decay": 1e-4},
                },
                "lr_scheduler": {
                    "class_path": "WarmUpCosineAnnealingScheduler",
                    "init_args": {
                        "warmup_epochs": 5,
                        "cosine_epochs": 15,
                    },
                },
                
                "model": lazy_instance(
                    LongTermForecastingExp, criterion="torch.nn.L1Loss", architecture=lazy_instance(BaseModel)
                ),
                "model_checkpoint.monitor": "val/loss",
                "model_checkpoint.monitor": "val/MeanSquaredError",
                "model_checkpoint.save_top_k": 2,
                "model_checkpoint.mode": "min",
                "model_checkpoint.auto_insert_metric_name": False,
                "model_checkpoint.dirpath": "checkpoints/",  # This will be overridden by LoggerSaveConfigCallback
                "model_checkpoint.filename": "epoch={epoch}-val_loss={val/loss:.2f}",
                "lr_monitor.logging_interval": "epoch",
                # "swa.swa_lrs": 1e-2,
                # "swa.swa_epoch_start": 6,
                # "swa.annealing_epochs": 20
            }
        )

        parser.link_arguments(
            source="data.enc_in",
            target="model.init_args.architecture.init_args.enc_in",
            apply_on="instantiate",
        )
        # if parser.get_value("model.task") != Task.CLASSIFICATION:
        parser.link_arguments(
            source="data.pred_len",
            target="model.init_args.architecture.init_args.pred_len",
            apply_on="instantiate",
        )
        parser.link_arguments(
            source="data.seq_len",
            target="model.init_args.architecture.init_args.seq_len",
            apply_on="instantiate",
        )

   
    def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
        """Overrides the model's :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` method if a
        single optimizer and optionally a scheduler argument groups are added to the parser as 'AUTOMATIC'."""
        if not self.auto_configure_optimizers:
            return

        parser = self._parser(subcommand)

        def get_automatic(
            class_type: Union[Type, Tuple[Type, ...]], register: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]]
        ) -> List[str]:
            automatic = []
            for key, (base_class, link_to) in register.items():
                if not isinstance(base_class, tuple):
                    base_class = (base_class,)
                if link_to == "AUTOMATIC" and any(issubclass(c, class_type) for c in base_class):
                    automatic.append(key)
            return automatic

        optimizers = get_automatic(Optimizer, parser._optimizers)
        lr_schedulers = get_automatic(LRSchedulerTypeTuple, parser._lr_schedulers)

        if len(optimizers) == 0:
            return

        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(
                f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model` expects at most one optimizer "
                f"and one lr_scheduler to be 'AUTOMATIC', but found {optimizers + lr_schedulers}. In this case the "
                "user is expected to link the argument groups and implement `configure_optimizers`, see "
                "https://lightning.ai/docs/pytorch/stable/common/lightning_cli.html"
                "#optimizers-and-learning-rate-schedulers"
            )

        optimizer_class = parser._optimizers[optimizers[0]][0]
        optimizer_init = self._get(self.config_init, optimizers[0])
        if not isinstance(optimizer_class, tuple):
            optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
        if not optimizer_init:
            # optimizers were registered automatically but not passed by the user
            return

        lr_scheduler_init = None
        if lr_schedulers:
            lr_scheduler_class = parser._lr_schedulers[lr_schedulers[0]][0]
            lr_scheduler_init = self._get(self.config_init, lr_schedulers[0])
            if not isinstance(lr_scheduler_class, tuple):
                lr_scheduler_init = _global_add_class_path(lr_scheduler_class, lr_scheduler_init)

        if is_overridden("configure_optimizers", self.model):
            _warn(
                f"`{self.model.__class__.__name__}.configure_optimizers` will be overridden by "
                f"`{self.__class__.__name__}.configure_optimizers`."
            )
        optimizer = instantiate_class(filter(lambda p: p.requires_grad, self.model.parameters()), optimizer_init)
        lr_scheduler = instantiate_class(optimizer, lr_scheduler_init) if lr_scheduler_init else None
        fn = partial(self.configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)
        update_wrapper(fn, self.configure_optimizers)  # necessary for `is_overridden`
        # override the existing method
        self.model.configure_optimizers = MethodType(fn, self.model)
