import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.fc_real = nn.Linear(input_dim, output_dim)
        self.fc_imag = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_real = torch.real(x)
        x_imag = torch.imag(x)
        out_real = self.fc_real(x_real) - self.fc_imag(x_imag)
        out_imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return torch.complex(out_real, out_imag)
