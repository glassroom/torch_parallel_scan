# coding: utf-8
from setuptools import setup

setup(name='torch_parallel_scan',
    version='1.0.0',
    description='Parallel scan over tensors of any shape for PyTorch.',
    url='https://github.com/glassroom/torch_parallel_scan',
    author='Franz A. Heinsen',
    author_email='franz@glassroom.com',
    license='MIT',
    packages=['torch_parallel_scan'],
    install_requires='torch',
    zip_safe=False)
