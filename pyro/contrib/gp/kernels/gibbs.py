from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel

def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


class Gibbs(Kernel):
    """
    :param function lengthscale_fn: Function to compute length scale for each dimension
    :param function ls_args: Extra args to be passed to `lengthscale` function
    """
    def __init__(self, input_dim, lengthscale_fn, args=None, active_dims=None):
        super(Gibbs, self).__init__(input_dim, active_dims)
        self.lengthscale_fn = _handle_args(lengthscale_fn, args)
        self.args = args
        assert callable(lengthscale_fn), 'Gibbs kernel requires a callable lengthscale function'

    def _square_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        X2 = (X ** 2).sum(1, keepdim=True)
        Z2 = (Z ** 2).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        return self.variance.expand(X.size(0))

    def forward(self, X, Z=None, diag=False):
        dim = len(self.active_dims)
        if diag:
            raise NotImplementedError

        if Z is None:
            Z = X

        rZ = self.lengthscale_fn(Z, self.args)
        rX = self.lengthscale_fn(X, self.args)
        r2 = self._square_dist(X,Z)

        rX2 = (rX**2).reshape(-1, 1, dim)
        rZ2 = (rZ**2).reshape(1, -1, dim)
        rX2_plus_rZ2 = torch.sum(rX2 + rZ2, dim=-1)

        return ( _torch_sqrt( torch.pow( (2.0 * rX.matmul(rZ.t())) / rX2_plus_rZ2, dim ) ) \
            * torch.exp( -1.0 * r2 / rX2_plus_rZ2 ) )

def _handle_args(func, args):
    def f(x, args):
        if args is None:
            return func(x)
        else:
            if not isinstance(args, tuple):
                args = (args,)
            return func(x, *args)
    return f