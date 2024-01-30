from torch import nn
from torch import Tensor


class GRUPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 16, dropout: float = 0.):
        super().__init__()

        self.gru = nn.GRU(11, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, tensor_in: Tensor, **kwargs):
        output, _ = self.gru(tensor_in)

        return self.linear(self.dropout(output))
