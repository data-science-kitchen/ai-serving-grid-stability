from torch import nn
from torch import Tensor


class GRUPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 16, dropout: float = 0.):
        super().__init__()

        self.gru = nn.GRU(11, 16, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, 1)

    def forward(self, tensor_in: Tensor, **kwargs):
        output, _ = self.gru(tensor_in)

        return self.linear(output)
