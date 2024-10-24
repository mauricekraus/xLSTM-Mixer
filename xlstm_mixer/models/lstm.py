from torch import nn
from einops import rearrange, reduce, repeat, pack, unpack
from .base_model import BaseModel



class LSTM(BaseModel):

    def __init__(
        self,
        pred_len: int,
        seq_len: int,
        enc_in: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        
    ) -> None:
        super().__init__(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(self.enc_in, embedding_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, enc_in*pred_len)



    def forecast(self, x_enc, x_mark_enc):
        out, _ = self.lstm(x_enc)
        out = out[:, -1, :]  # Get the output of the last time step

        out = self.fc(out)
        out = out.view(out.size(0), self.pred_len, -1)  # Reshape to [batch_size, prediction_horizon, num_features]

        return out
