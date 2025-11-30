import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """

    Supports:
      - logic networks  (logic=True)
      - neural actor-critic networks
      - optional softmax/sigmoid

    """

    def __init__(
            self,
            device,
            has_softmax=False,
            has_sigmoid=False,
            out_size=3,
            as_dict=False,
            logic=False,
    ):
        super().__init__()

        self.device = device
        self.logic = logic
        self.has_softmax = has_softmax
        self.has_sigmoid = has_sigmoid
        self.out_size = out_size
        self.as_dict = as_dict

        # ----------------------------------------------
        # Input dimensions
        # ----------------------------------------------
        self.logic_input_dim = 32       # 8*4
        self.neural_input_dim = 6 * 6 * 3   # 108

        input_dim = self.logic_input_dim if logic else self.neural_input_dim

        # ----------------------------------------------
        # Network architecture
        # ----------------------------------------------
        hidden_dim = 128

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, out_size)
        ).to(device)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, state):
        """
        state:
            Logic:  (batch,8,4)
            Neural: (batch,6,6,3)

        Output:
            (batch, out_size)
        """

        if self.logic:
            x = state.view(state.size(0), -1).float()
        else:
            # (batch,6,6,3) → (batch,108)
            x = state.view(state.size(0), -1).float()

        y = self.mlp(x)

        if self.has_softmax:
            y = F.softmax(y, dim=-1)

        if self.has_sigmoid:
            y = torch.sigmoid(y)

        return y.to(self.device)
