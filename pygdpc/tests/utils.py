import torch
import numpy as np


def gen_data(seed, k, num_series, num_periods):
    torch.manual_seed(seed)
    data = torch.randn(num_periods, num_series)
    f_true = torch.randn(num_periods + 1)
    for m in range(num_series):
        data[:, m] = f_true[:num_periods] * np.sin(2 * np.pi * m / num_series) \
                     + f_true[1:(num_periods + 1)] * (m / num_series) + 0.5 * torch.randn(num_periods)
    return data
