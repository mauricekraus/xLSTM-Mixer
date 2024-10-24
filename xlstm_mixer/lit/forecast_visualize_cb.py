import os
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from matplotlib import pyplot as plt
import torch
import numpy as np
from xlstm_mixer.lit.enums import ForecastingTaskOptions


class ForecastVisualizeCallback(Callback):

    def __init__(self, task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE, pred_len: int= 0, idxs: range = range(0,4), freq_epoch: int = 4 ) -> None:

        self.task_options = task_options
        self.pred_len = pred_len
        self.idxs = idxs
        self.freq_epoch = freq_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.freq_epoch != 0:
            return

        # make grid of predictions/ground truth
        batch_x = []
        batch_y = []
        batch_x_mark = []
        batch_y_mark = []
        for i in self.idxs:
            elem_x,elem_y,elem_x_mark, elem_y_mark = trainer.datamodule.train_dataset[i]

            batch_x.append(elem_x)
            batch_y.append(elem_y)
            batch_x_mark.append(elem_x_mark)
            batch_y_mark.append(elem_y_mark)
        
        batch_x = torch.from_numpy(np.stack(batch_x)).float().to(pl_module.device)
        batch_y = torch.from_numpy(np.stack(batch_y)).float().to(pl_module.device)
        batch_x_mark = torch.from_numpy(np.stack(batch_x_mark)).float().to(pl_module.device)
        batch_y_mark = torch.from_numpy(np.stack(batch_y_mark)).float().to(pl_module.device)
        
        pl_module.eval()
        with torch.no_grad():
            dec_input = None
            outputs = pl_module.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

            f_dim = -1 #-1 if self.task_options == ForecastingTaskOptions.MULTIVARIATE_2_UNIVARIATE else 0
            outputs = outputs[:, -self.pred_len:, f_dim:].contiguous().detach().cpu().numpy()
            batch_y = batch_y[:, -self.pred_len:, f_dim:].contiguous().detach().cpu().numpy()
            
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            
            for i, ax in enumerate(axs.flatten()):
                ax.plot(outputs[i], label='pred')
                ax.plot(batch_y[i], label='true')
                ax.legend()
            
            os.makedirs('pics', exist_ok=True)
            plt.savefig(f'pics/epoch_{trainer.current_epoch}.pdf', bbox_inches='tight')
        pl_module.train()

