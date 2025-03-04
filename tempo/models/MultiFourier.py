import numpy as np
import torch


class MultiFourier(torch.nn.Module):
    def __init__(self, N, P):
        super(MultiFourier, self).__init__()
        self.N = N
        self.P = P
        self.a = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)

    def forward(self, t):
        output = torch.zeros_like(t)

        # Shape: (batch_size, seq_len, max(N))
        t = t.unsqueeze(-1).repeat(1, 1, max(self.N))

        # Shape: (1, 1, max(N))
        n = torch.arange(max(self.N)).unsqueeze(0).unsqueeze(0)

        # Loop over seasonal components
        for j in range(len(self.N)):
            # Shape: (batch_size, seq_len, N[j])
            cos_terms = torch.cos(
                2 * np.pi * (n[..., : self.N[j]] + 1) * t[..., : self.N[j]] / self.P[j]
            )

            # Shape: (batch_size, seq_len, N[j])
            sin_terms = torch.sin(
                2 * np.pi * (n[..., : self.N[j]] + 1) * t[..., : self.N[j]] / self.P[j]
            )

            # Shape: (batch_size, seq_len, N[j])
            output += torch.matmul(cos_terms, self.a[: self.N[j], j]) + torch.matmul(
                sin_terms, self.b[: self.N[j], j]
            )

        return output
