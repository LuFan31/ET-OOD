from typing import Any

import torch
import torch.nn as nn
import numpy as np


class Tenergyprocessor:
    @torch.no_grad()
    def __call__(self, net: nn.Module, data: Any,):
        temperature = 1600
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        energy =torch.logsumexp(output, dim=1)
        Tenergy = temperature * torch.logsumexp(output / temperature, dim=1)

        return pred, Tenergy, energy
