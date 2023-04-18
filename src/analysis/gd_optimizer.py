from typing import Callable, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.optimizer import _use_grad_for_differentiable

from src.algorithms.gradient_descent_template import GD
from torch.optim import Optimizer

from src.models.logistic_regression import gradient


class GDOptimizer(Optimizer):
    """
    Adapts Torch's interface to work with our GD definition.
    Torch's approach uses Tensors. Default mathematical operations and numpy operations should be compatible
    """
    def __init__(self, params: Iterator[Parameter], gd: GD = None, gd_creator: Callable = None) -> None:
        super().__init__(params, {"differentiable": False})
        if gd:
            self.gd = gd
        else:
            self.gd = []
            self._gd_creator = gd_creator

    @_use_grad_for_differentiable
    def step(self, closure=None, loss=None):
        self.loss =loss
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            for i, param in enumerate(params_with_grad):
                if self._gd_creator:
                    if len(self.gd) > i:
                        gd = self.gd[i]
                    else:
                        gd = self._gd_creator(param)
                        self.gd.append(gd)
                else:
                    gd = self.gd

                # TODO: Does not work with nesterov as nesterov modifies w resulting in the gradient not existing anymore.
                # Need to find a way to recalucalte
                res = gd.step(param, lambda w: param.grad)
                param.copy_(res)  # Hack to do inplace replacement.
