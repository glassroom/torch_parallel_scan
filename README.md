# torch_parallel_scan

A simple implementation of parallel scan over sequences of tensors for PyTorch.

Only 40 lines of Python code, excluding docstrings and whitespace.


## Installing

```
pip install git+https://github.com/glassroom/torch_parallel_scan
```

Alternatively, you can download a single file to your project directory: [torch_parallel_scan.py](torch_parallel_scan/torch_parallel_scan.py).

The only dependency is PyTorch.


## Sample Usage


### Parallel Prefix Scan

```python
import torch
import torch_parallel_scan as tps

n, d = (100, 1024)
x = torch.randn(n, d, d) / (d**0.5)           # n square matrices
y = tps.prefix_scan(x, torch.matmul, dim=-3)  # cumulative matmul
```

### Parallel Reduce Scan

```python
import torch
import torch_parallel_scan as tps

n, d = (100, 1024)
x = torch.randn(n, d, d) / (d**0.5)           # n square matrices
y = tps.reduce_scan(x, torch.matmul, dim=-3)  # matmul of all matrices
```

## Notes

For both `prefix_scan` and `reduce_scan`, the binary associative function you pass as an argument must compute outputs that have the same shape as the inputs. If you wish to compute parallel scans over different shapes (e.g., products of matrices of different shapes), use padding. We have no plans to change this, because it would likely make the code in this repository significantly more complex.

We want the code in this repository to remain as short and simple as possible, so others can more easily understand and modify it for their own purposes.


## Citing

If our work is helpful to your research, please cite it:

```
@misc{heinsen2024torchparallelscan,
    title={An Implementation of Parallel Scan},
    author={Franz A. Heinsen},
    year={2024},
    primaryClass={cs.LG}
}
```

## How is this used at GlassRoom?

We conceived and implemented this code to be a component (e.g., part of a layer) of larger models that are in turn part of our AI software, nicknamed Graham. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code and release it as stand-alone open-source software without having to disclose any key intellectual property. We hope others find our work and our code useful.


