import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GDPC(nn.Module):

    def __init__(self, k, batch_size, epochs, lr=0.01, tol=1e-4, rescale=True):
        super().__init__()
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.mse = None
        self.component = None
        self.beta = None
        self.alpha = None
        self.conv = False
        self.niter = 0
        self.num_periods = None
        self.num_series = None
        self.rescale = rescale

    def forward(self, periods):
        fits = torch.zeros(len(periods), self.num_series, requires_grad=False)
        for t in range(len(periods)):
            for h in range(self.k + 1):
                fits[t, :] += self.component[periods[t] + h] * self.beta[:, h]
            fits[t, :] += self.alpha
        return fits

    @staticmethod
    def reconstruction_error(fitted, data):
        output = nn.MSELoss()
        rec = output(fitted, data)
        return rec

    def fit(self, data):
        self.num_periods, self.num_series = data.shape
        self.component = torch.randn(self.num_periods + self.k, requires_grad=True)
        self.beta = torch.randn(self.num_series, self.k + 1, requires_grad=True)
        self.alpha = torch.randn(self.num_series, requires_grad=True)
        opt = optim.Adam([self.component, self.alpha, self.beta], lr=self.lr)
        self.train()

        loss_new = self._train_epoch(data=data, opt=opt, batch_size=self.batch_size)
        self.niter += 1
        crit = self.tol + 1

        while self.niter < self.epochs and crit > self.tol:
            loss_old = loss_new
            loss_new = self._train_epoch(data=data, opt=opt, batch_size=self.batch_size)
            self.niter += 1
            crit = np.abs((loss_old - loss_new)/loss_old)

        self.eval()
        self.beta = self.beta.detach().numpy()
        self.alpha = self.alpha.detach().numpy()
        self.component = self.component.detach().numpy()
        self.mse = self.reconstruction_error(self.forward(range(self.num_periods)), data).item()
        if crit <= self.tol:
            self.conv = True

    # def _rescale(self):
    #     comp_mean = torch.mean(self.component)
    #     comp_std = torch.std(self.component)
    #     self.component = (self.component - comp_mean) / comp_std
    #     for h in range(self.k + 1):
    #         self.alpha = self.alpha + (comp_mean / comp_std) * self.beta[:, h]
    #     self.beta = self.beta * comp_std

    def _train_epoch(self, data, opt, batch_size):
        torch.autograd.set_detect_anomaly(True)
        total_loss = 0
        split = int(np.floor(self.num_periods/batch_size))
        batches = [list(range(num_batch * batch_size, (num_batch + 1) * batch_size)) for num_batch in range(split)]
        if split * batch_size < self.num_periods:
            batches.append(list(range(split * batch_size, self.num_periods)))
        for batch in batches:
            opt.zero_grad()
            fitted = self.forward(periods=batch)
            loss = self.reconstruction_error(fitted, data[batch, :])
            loss.backward()
            opt.step()
            total_loss += loss.item()
        return total_loss


