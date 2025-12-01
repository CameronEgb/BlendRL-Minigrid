import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsfr.utils.torch import softor, weight_sum


def init_identity_weights(X, device):
    ones = torch.ones((X.size(0),), dtype=torch.float32) * 100
    return torch.diag(ones).to(device)


class InferModule(nn.Module):
    """
    A class of differentiable foward-chaining inference.
    """

    def __init__(self, I, m, infer_step, gamma=0.01, device=None, train=False, clauses=None, atoms=None): # Added atoms argument
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(InferModule, self).__init__()
        self.I = I
        self.infer_step = infer_step
        # m is num of clauses
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        self.clauses = clauses
        self.atoms = atoms # Store atoms

        if not train:
            self.W = self.init_identity_weights(device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0))))).to(device)
        # clause functions
        self.cs = [ClauseFunction(i, I, gamma=gamma)
                   for i in range(self.I.size(0))]

    def init_identity_weights(self, device):
        ones = torch.ones((self.C,), dtype=torch.float32) * 10
        return torch.diag(ones).to(device)

    def forward(self, x):
        R = x
        for t in range(self.infer_step):
            R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        return R

    def r(self, x):
        B = x.size(0)  # batch size
        C = torch.stack([self.cs[i](x)
                         for i in range(self.I.size(0))], 0)
        
        W_star = torch.softmax(self.W, 1)
        # --- NEW DEBUG PRINT FOR CLAUSE WEIGHTS ---
        # print("DEBUG: InferModule.r - Softmaxed Clause Weights (W_star):")
        # for i in range(self.C): # Iterate through clauses
        #     clause_str = str(self.clauses[i]) if self.clauses else f"Clause {i}"
        #     for m_idx in range(self.m): # For each 'm' clause from the training setup
        #         weight_val = W_star[m_idx, i].item() # Access specific weight
        #         print(f"  {clause_str} (head={self.clauses[i].head.pred.name if self.clauses else 'N/A'}) - W_star[{m_idx},{i}]: {weight_val:.3f}")
        # --- END NEW DEBUG PRINT ---

        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        H = torch.sum(W_tild * C_tild, dim=1)

        # --- NEW DEBUG PRINT FOR H TENSOR ---
        # print(f"DEBUG: InferModule.r - H tensor (weighted sum) shape: {H.shape}")
        # if self.atoms: # Check if atoms are available
        #     from nsfr.fol.logic import Const # Import Const for debugging here
        #     from nsfr.utils.logic import get_index_by_predname # Import helper
        #     action_preds = ['up_to_goal', 'down_to_goal', 'right_to_goal', 'left_to_goal']
        #     for pred_name in action_preds:
        #         try:
        #             atom_idx = get_index_by_predname(pred_str=pred_name, atoms=self.atoms, args=[Const('img', dtype='image')])
        #             h_val = H[:, :, atom_idx].detach().cpu().numpy()
        #             print(f"DEBUG: InferModule.r - H['{pred_name}(img)'] for each batch: {h_val}")
        #         except ValueError as e:
        #             print(f"DEBUG: Could not find {pred_name}(img) in atoms. Error: {e}")
        # --- END NEW DEBUG PRINT ---

        R = softor(H, dim=0, gamma=self.gamma)
        return R

    def get_params(self):
        return self.W


class ClauseBodyInferModule(nn.Module):
    """
    A class of differentiable foward-chaining inference.
    """

    def __init__(self, I, gamma=0.01, device=None, train=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ClauseBodyInferModule, self).__init__()
        self.I = I
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train

        # clause functions
        # self.cs_bs = [ClauseBodySumFunction(I[i], I, gamma=gamma)
        #               for i in range(self.I.size(0))]
        self.cs_bs = [ClauseBodySumFunction(i, I, gamma=gamma)
                      for i in range(self.I.size(0))]
        self.cs = [ClauseFunction(i, I, gamma=gamma)
                   for i in range(self.I.size(0))]

    def init_identity_weights(self, device):
        ones = torch.ones((self.C,), dtype=torch.float32) * 100
        return torch.diag(ones).to(device)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # return self.r(x)
        return self.r_cb(x)

    def r(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        C = torch.stack([self.cs[i](x)
                         for i in range(self.I.size(0))], 0)
        return C

    def r_cb(self, x):
        # x: C * B * G
        B = x.size(1)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # infer from i-th valuation tensor using i-th clause
        C = torch.stack([self.cs_bs[i](x)
                         for i in range(self.I.size(0))], 0)
        return C

    def get_params(self):
        return self.W


class ClauseInferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk=None):
        """
        Infer module using each clause.
        The result is not amalgamated in terms of clauses.
        """
        super(ClauseInferModule, self).__init__()
        self.I = I
        self.I_bk = I_bk
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0))))).to(device)
        # clause functions
        self.cs = [ClauseFunction(I[i], I, gamma=gamma)
                   for i in range(self.I.size(0))]

        self.cs_bs = [ClauseBodySumFunction(I[i], I, gamma=gamma)
                      for i in range(self.I.size(0))]

        if not self.I_bk is None:
            self.cs_bk = [ClauseFunction(I_bk[i], I, gamma=gamma)
                          for i in range(self.I_bk.size(0))]

        if not I_bk is None:
            self.W_bk = init_identity_weights(I_bk, device)

        assert m == self.C, "Invalid m and C: " + \
                            str(m) + ' and ' + str(self.C)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.r_cb(x)

    def r_cb(self, x):
        # x: C * B * G
        B = x.size(1)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # infer from i-th valuation tensor using i-th clause
        C = torch.stack([self.cs_bs[i](x)
                         for i in range(self.I.size(0))], 0)
        return C

    def get_params(self):
        return self.W


class ClauseFunction(nn.Module):
    """
    A class of the clause function.
    """

    def __init__(self, i, I, gamma=0.01):
        super(ClauseFunction, self).__init__()
        self.i = i  # clause index
        self.I = I  # index tensor C * S * G, S is the number of possible substituions
        self.L = I.size(-1)  # number of body atoms
        self.S = I.size(-2)  # max number of possible substitutions
        self.gamma = gamma

    def forward(self, x):
        batch_size = x.size(0)  # batch size
        # B * G
        V = x
        # G * S * b
        I_i = self.I[self.i, :, :, :]

        # B * G -> B * G * S * L
        V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
        # G * S * L -> B * G * S * L
        I_i_tild = I_i.repeat(batch_size, 1, 1, 1)
        
        # --- NEW DEBUG PRINT IN CLAUSEFUNCTION.FORWARD ---
        # Only print for the first few clauses to avoid too much spam
        # if self.i < 4: # Assuming your action clauses are the first 4
        #     gathered_values = torch.gather(V_tild, 1, I_i_tild)
        #     product_values = torch.prod(gathered_values, 3)
        #     # Make sure this softor is not double-applied or removed
        #     clause_output_C = softor(product_values, dim=2, gamma=self.gamma) 
            
        #     print(f"DEBUG: ClauseFunction {self.i} - V_tild shape: {V_tild.shape}")
        #     print(f"DEBUG: ClauseFunction {self.i} - I_i_tild shape: {I_i_tild.shape}")
        #     print(f"DEBUG: ClauseFunction {self.i} - gathered_values max: {gathered_values.max().item():.3f}, min: {gathered_values.min().item():.3f}")
        #     print(f"DEBUG: ClauseFunction {self.i} - product_values max: {product_values.max().item():.3f}, min: {product_values.min().item():.3f}")
        #     print(f"DEBUG: ClauseFunction {self.i} - clause_output_C max: {clause_output_C.max().item():.3f}, min: {clause_output_C.min().item():.3f}")
        # --- END NEW DEBUG PRINT ---

        # B * G
        C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3),
                   dim=2, gamma=self.gamma)
        return C


class ClauseBodySumFunction(nn.Module):
    """
    A class of the clause function.
    """

    def __init__(self, i, I, gamma=0.01):
        super(ClauseBodySumFunction, self).__init__()
        self.i = i  # clause index
        self.I = I  # index tensor C * S * G, S is the number of possible substituions
        self.L = I.size(-1)  # number of body atoms
        self.S = I.size(-2)  # max number of possible substitutions
        self.gamma = gamma

    def forward(self, x):
        batch_size = x.size(0)  # batch size
        # B * G
        V = x
        # G * S * b
        I_i = self.I[self.i, :, :, :]

        # B * G -> B * G * S * L
        V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
        # G * S * L -> B * G * S * L
        I_i_tild = I_i.repeat(batch_size, 1, 1, 1)

        # B * G
        C = torch.sum(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3), dim=2)
        return C