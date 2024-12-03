# coding: utf-8

import torch
import math


def prefix_scan(x, prefix_func, dim, pad_value=0):
    """
    Apply prefix func in parallel over sequence, left to right.

    Args:
        x: tensor of shape [*preceding_dims, seq_len, *operand_dims].
        prefix_func: broadcastable binary associative function.
        dim: dimension over which to compute the parallel scan.
        pad_value: for padding sequences to a power of two. Default: 0.

    Output:
        y: tensor of shape [*preceding_dims, seq_len, *operand_dims].

    Sample use:
    >>> n, d = (100, 1024)
    >>> x = torch.randn(n, d, d) / (d**0.5)       # n square matrices
    >>> y = prefix_scan(x, torch.matmul, dim=-3)  # cumulative matmul
    """
    x = x.movedim(dim, -1)  # for easier indexing
    other_dims, seq_len = (x.shape[:-1], x.size(-1))
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    n_pads = 2 ** n_powers_of_2 - seq_len
    x = torch.nn.functional.pad(x, (0, n_pads), value=pad_value)
    for n in (2 ** torch.arange(n_powers_of_2)).tolist():
        x = x.view(*other_dims, -1, n * 2)
        last_on_L = x[..., (n - 1):n]
        last_on_L = last_on_L.movedim((-2, -1), (dim - 1, dim))
        all_on_R = x[..., n:]
        all_on_R = all_on_R.movedim((-2, -1), (dim - 1, dim))
        updated_on_R = prefix_func(last_on_L, all_on_R)
        updated_on_R = updated_on_R.movedim((dim - 1, dim), (-2, -1))
        x = torch.cat([x[..., :n], updated_on_R], dim=-1)
    x = x.view(*other_dims, -1)
    x = x[..., :seq_len]
    y = x.movedim(-1, dim)  # put dims back in orig order
    return y


def reduce_scan(x, reduce_func, dim):
    """
    Apply reduce_func in parallel over sequence, left to right.

    Args:
        x: tensor of shape [*preceding_dims, seq_len, *operand_dims].
        reduce_func: broadcastable binary associative function.
        dim: dimension over which to compute the parallel scan.

    Output:
        y: tensor of shape [*preceding_dims, *operand_dims].

    Sample use:
    >>> n, d = (100, 1024)
    >>> x = torch.randn(n, d, d) / (d**0.5)       # n square matrices
    >>> y = reduce_scan(x, torch.matmul, dim=-3)  # matmul of all matrices
    """
    x = x.movedim(dim, -1)  # for easier indexing
    other_dims, seq_len = (x.shape[:-1], x.size(-1))
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    for _ in range(n_powers_of_2):
        if x.size(-1) % 2 == 0:
            leftover = None
        else:
            leftover = x[..., -1:]
            x = x[..., :-1]
        x = x.view(*other_dims, -1, 2)
        operands_on_L = x[..., 0].movedim(-1, dim)
        operands_on_R = x[..., 1].movedim(-1, dim)
        x = reduce_func(operands_on_L, operands_on_R)
        x = x.movedim(dim, -1)
        if leftover is not None:
            x = torch.cat([x, leftover], dim=-1)
    y = x.squeeze(-1)
    return y
