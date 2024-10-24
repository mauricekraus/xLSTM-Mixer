import torch
from torch import nn
from einops import rearrange

class MultiScaleProcessor(nn.Module):
    def __init__(self, down_sampling_method: str, down_sampling_window: int, enc_in: int, down_sampling_layers: int):
        super().__init__()
        self.down_sampling_method = down_sampling_method
        self.down_sampling_window = down_sampling_window
        self.enc_in = enc_in
        self.down_sampling_layers = down_sampling_layers

        if self.down_sampling_method == "max":
            self.down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == "avg":
            self.down_pool = torch.nn.AvgPool1d(self.down_sampling_window, ceil_mode=False)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            self.down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            raise ValueError(f"Unsupported down sampling method: {self.down_sampling_method}")

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None):
        # B,T,C -> B,C,T
        x_enc = rearrange(x_enc, "b t v -> b v t")

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for _ in range(self.down_sampling_layers):
            x_enc_sampling = self.down_pool(x_enc_ori)

            x_enc_sampling_list.append(rearrange(x_enc_sampling, "b v t -> b t v"))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                # Adjust slicing to match avg_pool1d behavior
                slice_length = x_mark_enc_mark_ori.size(1) // self.down_sampling_window * self.down_sampling_window
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, :slice_length:self.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :slice_length:self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc
