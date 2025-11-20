import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    BlendRL-compatible MLP for MiniGrid (5x5 grid).

    Supports:
      - logic networks  (logic=True)
      - neural actor-critic networks
      - optional softmax/sigmoid

    Input shapes:
      Logic state:  (batch, 25, 1)
      Neural state: (batch, 1, 5, 5, 3)
    """

    def __init__(
            self,
            device,
            has_softmax=False,
            has_sigmoid=False,
            out_size=7,
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
        self.logic_input_dim = 20       # 4*4
        self.neural_input_dim = 4 * 84 * 84   #4 * 84 * 84

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
            Logic:  (batch,25,1)
            Neural: (batch,1,5,5,3)

        Output:
            (batch, out_size)
        """

        if self.logic:
            x = state.view(state.size(0), -1).float()
        else:
            # (batch,1,5,5,3) → (batch,75)
            x = state.view(state.size(0), -1).float()

        y = self.mlp(x)

        if self.has_softmax:
            y = F.softmax(y, dim=-1)

        if self.has_sigmoid:
            y = torch.sigmoid(y)

        return y.to(self.device)
