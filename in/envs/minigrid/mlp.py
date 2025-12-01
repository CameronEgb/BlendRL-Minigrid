import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """

    Supports:
      - logic networks  (logic=True)
      - neural actor-critic networks (now a CNN)
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

        if self.logic:
            # ----------------------------------------------
            # Input dimensions
            # ----------------------------------------------
            self.logic_input_dim = 32       # 8*4
            input_dim = self.logic_input_dim

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
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ).to(device)
            
            self.fc = nn.Sequential(
                nn.Linear(32 * 2 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, out_size)
            ).to(device)


    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, state):
        """
        state:
            Logic:  (batch,8,4)
            Neural: (batch,8,8,3)

        Output:
            (batch, out_size)
        """

        if self.logic:
            x = state.view(state.size(0), -1).float()
            y = self.mlp(x)
        else:
            # (batch,8,8,3) -> (batch,3,8,8)
            x = state.permute(0, 3, 1, 2).float()
            x = self.cnn(x)
            x = x.reshape(x.size(0), -1) # flatten
            y = self.fc(x)

        if self.has_softmax:
            y = F.softmax(y, dim=-1)

        if self.has_sigmoid:
            y = torch.sigmoid(y)

        return y.to(self.device)