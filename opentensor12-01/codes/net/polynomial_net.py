import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PolynomialNet(nn.Module):
    """
    Minimal policy/value network for polynomial circuit environment.

    Inputs:
        tensors: [B, T, S, S, S] float
        scalars: [B, 3] float (step counts; currently unused but kept for API symmetry)
        mask:    [B, A] bool for valid actions

    Outputs (train mode):
        logits: [B, A] masked logits
        values: [B] scalar value estimates

    Outputs (infer mode):
        actions: [B] sampled valid action indices
        probs:   [B, A] probability distribution over actions
        values:  [B]
    """

    def __init__(self, action_dim: int, hidden_dim: int = 256, T: int = 1, s_size: int = 4, device: str = "cpu"):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.T = T
        self.S = s_size

        input_dim = T * (s_size ** 3)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.mode = "train"
        self.hint_actions = None  # Optional list of preferred actions by step index

    def set_mode(self, mode: str):
        assert mode in ["train", "infer"]
        self.mode = mode

    def forward(self, tensors, scalars=None, mask: Optional[torch.Tensor] = None):
        if not torch.is_tensor(tensors):
            tensors = torch.from_numpy(tensors).float().to(self.device)
        else:
            tensors = tensors.float().to(self.device)
        batch_size = tensors.shape[0]
        if scalars is None:
            scalars_t = None
        elif not torch.is_tensor(scalars):
            scalars_t = torch.from_numpy(scalars).float().to(self.device)
        else:
            scalars_t = scalars.float().to(self.device)

        flat = tensors.view(batch_size, -1)
        feat = self.encoder(flat)
        logits = self.policy_head(feat)
        values = self.value_head(feat).squeeze(-1)

        if mask is not None:
            mask = mask.to(self.device)
            logits = logits.masked_fill(~mask, float("-inf"))

        # Optional hint: push known-good action for the current step.
        if self.hint_actions is not None and scalars_t is not None:
            step_ct = int(scalars_t[0, 0].item()) if scalars_t.ndim > 1 else int(scalars_t[0].item())
            if step_ct < len(self.hint_actions):
                hint_idx = self.hint_actions[step_ct]
                if mask is None or bool(mask.view(-1)[hint_idx]):
                    logits[:, hint_idx] = logits.max(dim=-1, keepdim=True).values.squeeze(-1) + 5.0

        if self.mode == "train":
            return logits, values

        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return actions, probs, values

    def policy(self, output):
        """Compatibility helper for existing call sites."""
        actions, probs, _ = output
        return actions.detach().cpu().numpy(), probs.detach().cpu().numpy()

    def value(self, output):
        if self.mode == "train":
            _, values = output
            vals = values.detach().cpu().numpy()
        else:
            _, _, values = output
            vals = values.detach().cpu().numpy()
        return vals, vals
